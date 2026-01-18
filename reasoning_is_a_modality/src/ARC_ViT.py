from typing import Optional, Tuple, List

from utils.pos_embed import VisionRotaryEmbeddingFast
import torch
from torch import nn

from timm.models.vision_transformer import PatchEmbed

import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        no_rope: int = 1,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        if self.head_dim % 2 != 0:
            raise ValueError("Rotary embeddings require the head dimension to be even")

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        half_head_dim = embed_dim // num_heads // 2
        self.rotary = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=int(max_seq_len ** 0.5),
            no_rope=no_rope,
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [B, S, D]
        key_padding_mask: [B, S] (True = pad)
        attn_mask: broadcastable to [B, 1, S, S] or [1, 1, S, S],
                   with 0 for allowed, -inf for disallowed.
        """
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)  # [B, S, 3*D]
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, Dh]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, S, Dh]

        q = self.rotary(q)
        k = self.rotary(k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, S, S]

        # Apply key padding mask (mask out keys)
        if key_padding_mask is not None:
            # key_padding_mask: [B, S] -> [B, 1, 1, S]
            mask = key_padding_mask[:, None, None, :].to(dtype=torch.bool)
            attn_scores = attn_scores.masked_fill(
                mask,
                torch.finfo(attn_scores.dtype).min,
            )

        # Apply structured attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                # True = disallowed
                attn_scores = attn_scores.masked_fill(
                    attn_mask, torch.finfo(attn_scores.dtype).min
                )
            else:
                # additive mask with 0 / -inf
                attn_scores = attn_scores + attn_mask


        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)  # [B, H, S, Dh]
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        context = self.proj(context)
        context = self.proj_dropout(context)
        return context


class ARCTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        max_seq_len: int,
        no_rope: int = 1,
        num_task_tokens: int = 1,
        num_ctx_tokens_max: int = 0,
        patch_local_attn: str = "8",   # NEW: "self" | "4" | "8"
    ) -> None:
        super().__init__()
        self.num_task_tokens = num_task_tokens
        self.num_ctx_tokens_max = num_ctx_tokens_max

        # --- Dense self-attention (everyone <-> everyone) ---
        self.self_attn_dense = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            no_rope=no_rope,
        )
        self.dropout_dense = nn.Dropout(dropout)
        self.norm_dense = nn.LayerNorm(embed_dim)

        # --- Structured self-attention (CTX+task hub) ---
        # Same dimensions as dense, but with an attention mask
        self.patch_local_attn = patch_local_attn
        # Precompute structured mask ONCE (bool), reused every forward
        struct_bool = self._precompute_struct_mask(max_seq_len=max_seq_len)
        self.register_buffer("struct_attn_mask_bool", struct_bool, persistent=False)

        self.self_attn_struct = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            no_rope=no_rope,
        )
        self.dropout_struct = nn.Dropout(dropout)
        self.norm_struct = nn.LayerNorm(embed_dim)

        # --- Shared MLP ---
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.activation = nn.GELU()
        self.dropout_mlp1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout_mlp2 = nn.Dropout(dropout)
        self.norm_mlp = nn.LayerNorm(embed_dim)

    def _precompute_struct_mask(self, max_seq_len: int) -> torch.Tensor:
        """
        Returns bool mask [1,1,S,S] where True means "disallow".
        Only depends on S=max_seq_len, prefix_len, patch_local_attn.
        """
        S = max_seq_len
        prefix_len = min(self.num_task_tokens + self.num_ctx_tokens_max, S)
        patch_start = prefix_len
        patch_len = S - patch_start

        # default: allow everything
        mask = torch.zeros((1, 1, S, S), dtype=torch.bool)

        if patch_len <= 0:
            return mask

        # disallow all patch->patch first
        mask[:, :, patch_start:, patch_start:] = True
        # none patch
        if self.patch_local_attn == "none":
            return mask  # keep patch->patch fully disallowed

        side = int(math.isqrt(patch_len))
        if side * side != patch_len:
            # fallback: allow self only
            diag = torch.arange(patch_len)
            mask[0, 0, patch_start + diag, patch_start + diag] = False
            return mask

        if self.patch_local_attn == "self":
            offsets = [(0, 0)]
        elif self.patch_local_attn == "4":
            offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        else:  # "8"
            offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]

        # unmask local neighbors
        for q in range(patch_len):
            r0, c0 = divmod(q, side)
            qi = patch_start + q

            for dr, dc in offsets:
                rr, cc = r0 + dr, c0 + dc
                if 0 <= rr < side and 0 <= cc < side:
                    kj = patch_start + (rr * side + cc)
                    mask[0, 0, qi, kj] = False

        return mask

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # ---- 1) Dense self-attention (pre-norm) ----
        x_norm = self.norm_dense(x)
        x_att = self.self_attn_dense(
            x_norm,
            key_padding_mask=key_padding_mask,
            attn_mask=None,
        )
        x = x + self.dropout_dense(x_att)
    
        # ---- 2) Structured self-attention (CTX hub, pre-norm) ----
        # CTX (and task) attend ALL; patches attend only to prefix
        x_norm = self.norm_struct(x)
        struct_mask = self.struct_attn_mask_bool
        # move to correct device lazily (buffer will move with model.to(device) anyway)

        x_att = self.self_attn_struct(
            x_norm,
            key_padding_mask=key_padding_mask,
            attn_mask=struct_mask,
        )
        x = x + self.dropout_struct(x_att)
    
        # ---- 3) MLP (pre-norm) ----
        x_norm = self.norm_mlp(x)
        x_mlp = self.linear1(x_norm)
        x_mlp = self.activation(x_mlp)
        x_mlp = self.dropout_mlp1(x_mlp)
        x_mlp = self.linear2(x_mlp)
        x = x + self.dropout_mlp2(x_mlp)
    
        return x



class ARCTransformerEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth: int,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        max_seq_len: int,
        no_rope: int = 0,
        num_task_tokens: int = 1,
        num_ctx_tokens_max: int = 0,
        patch_local_attn: str = "8",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ARCTransformerEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                    no_rope=no_rope,
                    num_task_tokens=num_task_tokens,
                    num_ctx_tokens_max=num_ctx_tokens_max,
                    patch_local_attn=patch_local_attn,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x

IGNORE_INDEX = 10

class ARCViT(nn.Module):
    """role split Vision Transformer tailored for ARC tasks with low channel controlled by high channel."""

    def __init__(
        self,
        num_tasks: int,
        image_size: int = 30,
        num_colors: int = 12,
        embed_dim: int = 256,
        depth: int = 10,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        num_task_tokens: int = 1,
        patch_size: int = 2,
        num_ctx_tokens: int = 4,  # capacity (P_max) for CTX tokens per sample
        num_recurrent_steps: int = 2, # number of recurrents 
        ema_alpha: float = 0.9,
        patch_local_attn: str = "8" # "self"|"4"|"8" 
    ) -> None:
        super().__init__()

        if image_size <= 0:
            raise ValueError("`image_size` must be > 0.")
        if num_colors <= 0:
            raise ValueError("`num_colors` must be > 0.")
        if num_tasks <= 0:
            raise ValueError("`num_tasks` must be > 0.")

        self.image_size = image_size
        self.num_colors = num_colors
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_task_tokens = num_task_tokens
        self.num_ctx_tokens_max = num_ctx_tokens
        self.patch_local_attn = patch_local_attn

        if patch_size is None:
            self.seq_length = image_size * image_size
        else:
            self.seq_length = (image_size // patch_size) ** 2

        print(
            f"Patch size: {self.patch_size}, sequence length: {self.seq_length}, "
            f"max ctx tokens: {self.num_ctx_tokens_max}"
        )

        # embeddings
        self.color_embed = nn.Embedding(num_colors, embed_dim)
        self.task_token_embed = nn.Embedding(num_tasks, embed_dim * self.num_task_tokens)
        self.patch_embed = PatchEmbed(
            image_size, patch_size, embed_dim, embed_dim, bias=True
        )

        # 2D positional embed only for patch tokens
        self.positional_embed = nn.Parameter(
            torch.zeros(1, self.seq_length, embed_dim)
        )

        # CTX encoder: Δ = E_y - E_x pooled over patches, then MLP
        self.ctx_norm = nn.LayerNorm(embed_dim)
        self.ctx_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # transformer encoder
        total_seq_len = self.num_task_tokens + self.num_ctx_tokens_max + self.seq_length
        self.encoder = ARCTransformerEncoder(
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            max_seq_len=total_seq_len,
            no_rope=self.num_task_tokens + self.num_ctx_tokens_max,
            num_task_tokens=self.num_task_tokens,
            num_ctx_tokens_max=self.num_ctx_tokens_max,
            patch_local_attn=self.patch_local_attn,
        )

        # GRU-style gate over depth: [h; hist] -> gate in [0,1] ---
        # Shape: (B, S, 2D) -> (B, S, D)
        self.num_recurrent_steps = num_recurrent_steps
        self.ema_alpha = ema_alpha
        self.recurrent_gate = nn.Linear(2 * embed_dim, embed_dim)
        # bias < 0 so gate starts near 0 (small updates at init)
        nn.init.constant_(self.recurrent_gate.bias, -1.0)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(
            embed_dim,
            num_colors * (1 if patch_size is None else patch_size) ** 2,
        )
        self._reset_parameters()
        # add a small embedding to determine stages
        self.step_embed = nn.Embedding(self.num_recurrent_steps, embed_dim)
        nn.init.normal_(self.step_embed.weight, std=0.02)

        # Correctness head: 1-layer linear -> logit
        # Used only in TTT as a training-time critic to push backbone gradient
        self.correct_head = nn.Linear(2 * embed_dim, 1)


    def _compute_correct_features(
        self,
        encoded: torch.Tensor,
        num_prefix: int,
    ) -> torch.Tensor:
        """
        encoded: [B, S, D] final encoded tokens (after recurrence, before head).
        num_prefix: number of prefix tokens (task + CTX).
        Returns:
            feats: [B, 2D] = [global_patch_feat, task_feat]
        """
        B, S, D = encoded.shape

        # patch tokens only
        patch_feats = encoded[:, num_prefix:, :]   # [B, L, D]
        global_feat = patch_feats.mean(dim=1)      # [B, D]

        # task token at index 0
        task_feat = encoded[:, 0, :]               # [B, D]

        feats = torch.cat([global_feat, task_feat], dim=-1)  # [B, 2D]
        return feats


        
    # GRU style Residual Recurrent Gated Transformer, also return the middle stage
    def _encode_with_recurrence(
        self,
        hidden_states: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_middle=False,
    ) -> torch.Tensor:
        """
        GRU-style recurrent depth around self.encoder.

        When num_recurrent_steps == 1, this reduces to a single call to self.encoder.
        For >1, we keep an EMA 'hist' and at each step:
            update = encoder(h + hist)
            gate   = σ(W [h, hist])
            h      = h + gate * update
            hist   = ema_alpha * hist + (1 - ema_alpha) * h
        """
        # no recurrence: original behavior
        if self.num_recurrent_steps <= 1:
            h = self.encoder(hidden_states, key_padding_mask=key_padding_mask)
            return (h, h) if return_middle else (h, None)

        h = hidden_states                     # [B, S, D]
        hist = h                              # start hist as copy of initial state
        scale_global = 1.0 / float(self.num_recurrent_steps)

        B, S, D = h.shape

        # build fixed prefix mask for CTX+task vs patch
        prefix_len = min(self.num_task_tokens + self.num_ctx_tokens_max, S)
        h_middle = None

        for i in range(self.num_recurrent_steps):
            step_emb = self.step_embed.weight[i].view(1, 1, -1)  # [1,1,D]
            h_step = h + step_emb

            # gate = σ(W [h, hist])
            z = torch.cat([h_step, hist], dim=-1)
            gate = torch.sigmoid(self.recurrent_gate(z))

            # single "cell" update: run the whole encoder as the transition f
            update = self.encoder(h_step + hist, key_padding_mask=key_padding_mask) # [B, S, D]

            # CTX/task vs patch scaling
            # task+CTX: scale_global * 1.0 ; patches: scale_global * 0.5, this unsymettric structure makes CTX/task token have more updates
            scale_vec = h.new_full((B, S, 1), scale_global * 0.5)
            scale_vec[:, :prefix_len, :] = scale_global * 1.0
            
            # residual GRU-style update with scale
            h = h + scale_vec * gate * update                         # [B, S, D]
            
            # store middle 
            if i == (self.num_recurrent_steps//2)-1:
                h_middle = h
            # update EMA history with new state
            hist = self.ema_alpha * hist + (1.0 - self.ema_alpha) * h

        return (h, h_middle) if return_middle else (h, None)

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.positional_embed, std=0.02)
        nn.init.trunc_normal_(self.task_token_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.color_embed.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def _encode_contexts(
        self,
        contexts: Optional[torch.Tensor],
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], int, Optional[torch.Tensor]]:
        """
        contexts: [B, P_max, 2, H, W] long tensor with 1 <= P_max <= num_ctx_tokens_max.
        Real ctx pairs are always valid; extra capacity is padded here.

        Returns:
        ctx_tokens: [B, P_cap, D]  where P_cap = self.num_ctx_tokens_max (STATIC)
        P_cap:      int
        ctx_pad_mask: [B, P_cap] bool, True = pad (ignored by attention & RoPE)
        """
        if contexts is None:
            raise ValueError("contexts must be provided (1–num_ctx_tokens_max pairs)")

        B, P_max, two, H, W = contexts.shape
        P_cap = self.num_ctx_tokens_max

        if P_max > P_cap:
            # Hard capacity clamp: drop extra pairs, or assert depending on your preference.
            contexts = contexts[:, :P_cap]
            P_max = P_cap

        # Build a fixed-size [B, P_cap, 2, H, W] tensor
        # Fill padded slots with IGNORE_INDEX so we can detect them.
        ctx_full = contexts.new_full((B, P_cap, 2, H, W), IGNORE_INDEX)
        ctx_full[:, :P_max] = contexts  # remaining slots stay as IGNORE_INDEX

        ctx_full = ctx_full.to(device).contiguous()  # [B, P_cap, 2, H, W]

        # detect which slots are real (not all IGNORE_INDEX)
        x_ctx = ctx_full[:, :, 0]  # [B, P_cap, H, W]
        y_ctx = ctx_full[:, :, 1]
        valid_x = (x_ctx != IGNORE_INDEX).any(dim=(-2, -1))
        valid_y = (y_ctx != IGNORE_INDEX).any(dim=(-2, -1))
        valid = valid_x | valid_y              # [B, P_cap] bool
        ctx_pad_mask = ~valid                  # True = padding

        # flatten pairs: [B * P_cap * 2, H, W]  (STATIC total length)
        ctx_flat = ctx_full.reshape(B * P_cap * 2, H, W)

        # reuse main embedding pipeline
        emb = self.color_embed(ctx_flat.long())           # [B*P_cap*2, H, W, D]
        emb = self.patch_embed(emb.permute(0, 3, 1, 2))   # [B*P_cap*2, L, D]
        emb = emb.reshape(B, P_cap, 2, -1, self.embed_dim)
        L = emb.size(3)  # number of patches per grid

        E_x = emb[:, :, 0]    # [B, P_cap, L, D]
        E_y = emb[:, :, 1]    # [B, P_cap, L, D]

        delta = E_y - E_x                     # [B, P_cap, L, D]
        pooled = delta.mean(dim=2)            # [B, P_cap, D]
        pooled = self.ctx_norm(pooled)
        ctx_vecs = self.ctx_mlp(pooled)       # [B, P_cap, D]

        # zero-out padded ctx slots (for cleanliness)
        ctx_vecs = ctx_vecs.masked_fill(ctx_pad_mask.unsqueeze(-1), 0.0)

        # IMPORTANT: P_eff is now STATIC = P_cap, to keep RoPE dimensions stable
        P_eff = P_cap

        return ctx_vecs, P_eff, ctx_pad_mask


    def forward(
        self,
        pixel_values: torch.Tensor,
        contexts: Optional[torch.Tensor],
        task_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        middle_loss: bool = False,
        return_encoded: bool = False,
    ) -> torch.Tensor:

        if pixel_values.dim() != 3:
            raise ValueError("`pixel_values` must be (batch, height, width).")
        if (
            pixel_values.size(1) != self.image_size
            or pixel_values.size(2) != self.image_size
        ):
            raise ValueError(
                "`pixel_values` height/width must match configured image_size="
                f"{self.image_size}. Received {pixel_values.shape[1:]}"
            )

        batch_size = pixel_values.size(0)
        device = pixel_values.device

        # ---- main patch tokens ----
        tokens = self.color_embed(pixel_values.long())         # [B, H, W, D]
        tokens = self.patch_embed(tokens.permute(0, 3, 1, 2))  # [B, L, D]
        tokens = tokens + self.positional_embed[:, : tokens.size(1), :]

        # ---- task tokens ----
        task_tokens = self.task_token_embed(task_ids.long())   # [B, num_task_tokens*D]
        task_tokens = task_tokens.reshape(batch_size, self.num_task_tokens, -1)

        # ---- CTX tokens ----
        ctx_tokens, num_ctx, ctx_pad_mask = self._encode_contexts(contexts, device)
        if ctx_tokens is not None and num_ctx > 0:
            hidden_states = torch.cat([task_tokens, ctx_tokens, tokens], dim=1)
            num_prefix = self.num_task_tokens + num_ctx
        else:
            hidden_states = torch.cat([task_tokens, tokens], dim=1)
            num_prefix = self.num_task_tokens
            ctx_pad_mask = None

        hidden_states = self.dropout(hidden_states)

        # ---- key padding mask (task + ctx + patch) ----
        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (
                batch_size,
                self.image_size,
                self.image_size,
            ):
                raise ValueError("`attention_mask` must match pixel grid size.")

            if self.patch_size is not None:
                attention_mask_p = attention_mask.reshape(
                    batch_size,
                    self.image_size // self.patch_size,
                    self.patch_size,
                    self.image_size // self.patch_size,
                    self.patch_size,
                )
                attention_mask_p = torch.max(
                    torch.max(attention_mask_p, dim=2)[0], dim=3
                )[0]  # [B, H_p, W_p]
            else:
                attention_mask_p = attention_mask

            flat_mask = attention_mask_p.view(batch_size, self.seq_length)
            pad_mask_patches = ~flat_mask.bool()  # True where we want to mask

            prefix_masks = [
                torch.zeros(
                    batch_size,
                    self.num_task_tokens,
                    device=device,
                    dtype=torch.bool,
                )
            ]
            if ctx_pad_mask is not None:
                prefix_masks.append(ctx_pad_mask)
            prefix_mask = torch.cat(prefix_masks, dim=1)  # [B, num_prefix]

            key_padding_mask = torch.cat(
                [prefix_mask, pad_mask_patches], dim=1
            )  # [B, num_prefix + L]

        # ---- transformer (with optional GRU recurrence) + head ----
        encoded, encoded_middle = self._encode_with_recurrence(
            hidden_states,
            key_padding_mask=key_padding_mask,
            return_middle=middle_loss,
        )
        encoded = self.norm(encoded)

        # calc middle logit if middle_loss is True
        if middle_loss:
            encoded_middle = self.norm(encoded_middle)
            pixel_states_middle = encoded_middle[:, num_prefix:, :]  # [B, L, D]
            logits_middle = self.head(pixel_states_middle)
            logits_middle = logits_middle.reshape(
                -1,
                self.image_size // self.patch_size,
                self.image_size // self.patch_size,
                self.patch_size,
                self.patch_size,
                self.num_colors,
            )
            logits_middle = logits_middle.permute(0, 1, 3, 2, 4, 5)
            logits_middle = logits_middle.reshape(
                batch_size, self.image_size, self.image_size, self.num_colors
            )
            logits_middle = logits_middle.permute(0, 3, 1, 2)
        
        # strip task+ctx tokens, keep patch positions only
        pixel_states = encoded[:, num_prefix:, :]  # [B, L, D]

        logits = self.head(pixel_states)
        logits = logits.reshape(
            -1,
            self.image_size // self.patch_size,
            self.image_size // self.patch_size,
            self.patch_size,
            self.patch_size,
            self.num_colors,
        )
        logits = logits.permute(0, 1, 3, 2, 4, 5)
        
        logits = logits.reshape(
            batch_size, self.image_size, self.image_size, self.num_colors
        )
        
        logits = logits.permute(0, 3, 1, 2)

        if middle_loss and return_encoded:
            return logits, logits_middle, encoded, num_prefix
        elif middle_loss:
            return logits, logits_middle
        elif return_encoded:
            return logits, encoded, num_prefix
        return logits