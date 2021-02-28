from delight.delight_transformer_layer import DeLighTTransformerEncoderLayer
from torch import nn
import torch
import numpy as np
from argparse import Namespace

default_arg = Namespace(
    act_type='swish', 
    activation_dropout=0.0, 
    adam_betas='(0.9, 0.98)', 
    adam_eps=1e-08, 
    adaptive_input=False, 
    adaptive_softmax_cutoff=None, 
    adaptive_softmax_dropout=0, 
    adaptive_softmax_factor=4, 
    all_gather_list_size=16384, 
    arch='delight_transformer_wmt14_en_de', 
    attention_dropout=0.03, 
    best_checkpoint_metric='loss', 
    bpe=None, 
    broadcast_buffers=False, 
    bucket_cap_mb=25, 
    clip_norm=0.0, 
    cpu=False, 
    criterion='label_smoothed_cross_entropy', 
    curriculum=0, 
    data='data-bin/wmt14_en_de', 
    dataset_impl=None, 
    ddp_backend='no_c10d', 
    decoder_learned_pos=False, 
    decoder_normalize_before=False, 
    define_iclr=False, 
    delight_dec_ffn_red=4,
    delight_dec_layers=8, 
    delight_dec_max_depth=8, 
    delight_dec_max_groups=8, 
    delight_dec_min_depth=4, 
    delight_dec_scaling='block', 
    delight_dec_width_mult=2.0, 
    delight_dropout=0.0, 
    delight_emb_depth=4, 
    delight_emb_dropout=0.1, 
    delight_emb_map_dim=128, 
    delight_emb_max_groups=8, 
    delight_emb_out_dim=256, 
    delight_emb_width_mult=2.0,
    delight_enc_ffn_red=4, 
    delight_enc_layers=8, 
    delight_enc_max_depth=8, 
    delight_enc_max_groups=8, 
    delight_enc_min_depth=4, 
    delight_enc_scaling='block', 
    delight_enc_width_mult=2.0, 
    device_id=0, 
    disable_validation=False, 
    distributed_backend='nccl', 
    distributed_init_method=None, 
    distributed_no_spawn=False, 
    distributed_port=50786, 
    distributed_rank=0, 
    distributed_world_size=1, 
    dropout=0.07, 
    empty_cache_freq=0, 
    encoder_learned_pos=False, 
    encoder_normalize_before=False, 
    eval_bleu=False, eval_bleu_args=None, 
    eval_bleu_detok='space', 
    eval_bleu_detok_args=None, 
    eval_bleu_print_samples=False, 
    eval_bleu_remove_bpe=None, 
    eval_tokenized_bleu=False, 
    fast_stat_sync=False, 
    ffn_dropout=0.07, 
    find_unused_parameters=False, 
    fix_batches_to_gpus=False, 
    fixed_validation_seed=None, 
    fp16=False, 
    fp16_init_scale=128, 
    fp16_no_flatten_grads=False, 
    fp16_scale_tolerance=0.0, 
    fp16_scale_window=None, 
    glt_shuffle=True, 
    keep_best_checkpoints=-1, 
    keep_interval_updates=-1, 
    keep_last_epochs=10, 
    label_smoothing=0.1, 
    left_pad_source='True', 
    left_pad_target='False', 
    load_alignments=False, 
    log_format=None, 
    log_interval=1000, 
    lr=[1e-07], 
    lr_period_updates=20000.0, 
    lr_scheduler='cosine', 
    lr_shrink=1.0, 
    max_epoch=0, 
    max_lr=0.007, 
    max_sentences=None, 
    max_sentences_valid=None, 
    max_source_positions=1024, 
    max_target_positions=1024, 
    max_tokens=4096, 
    max_tokens_valid=4096, 
    max_update=30000, 
    maximize_best_checkpoint_metric=False, 
    memory_efficient_fp16=False, 
    min_loss_scale=0.0001, 
    min_lr=1e-09, 
    no_epoch_checkpoints=False, 
    no_glt_shuffle=False, 
    no_last_checkpoints=False, 
    no_progress_bar=True, 
    no_save=False, 
    no_save_optimizer_state=False, 
    no_scale_embedding=False, 
    no_token_positional_embeddings=False, norm_type='ln', num_workers=1, optimizer='adam', optimizer_overrides='{}', patience=-1, pe_dropout=0.1, print_stats=False, required_batch_size_multiple=8, reset_dataloader=False, reset_lr_scheduler=False, reset_meters=False, reset_optimizer=False, restore_file='checkpoint_last.pt', save_dir='./results_wmt14_en2de/delight_out_256', save_interval=1, save_interval_updates=0, seed=1, sentence_avg=False, share_all_embeddings=True, share_decoder_input_output_embed=True, skip_invalid_size_inputs_valid_test=False, source_lang=None, src_len_ps=20, t_mult=1.0, target_lang=None, task='translation', tensorboard_logdir='', tgt_len_ps=20, threshold_loss_scale=None, tie_adaptive_proj=False, tie_adaptive_weights=False, tokenizer=None, train_subset='train', truncate_source=False, update_freq=[16], upsample_primary=1, use_bmuf=False, use_old_adam=False, user_dir=None, valid_subset='valid', validate_interval=1, warmup_init_lr=1e-07, warmup_updates=10000, weight_decay=0.0)


class DeLighTTransformer(nn.Module):
    def __init__(self, emb_dim, num_tokens, max_seq_len, args=default_arg):
        super().__init__()
        
        self.emb = nn.Embedding(num_tokens, emb_dim, padding_idx=0)
        self.pos_emb = nn.Parameter(torch.randn(max_seq_len, emb_dim))

        self.layers = nn.ModuleList()
        dextra_depths = np.linspace(start=args.delight_enc_min_depth,
                                         stop=args.delight_enc_max_depth,
                                         num=args.delight_enc_layers,
                                         dtype=np.int)

        depth_ratio = (args.delight_enc_max_depth * 1.0) / args.delight_enc_min_depth

        width_multipliers = np.linspace(start=args.delight_enc_width_mult,
                                    stop=args.delight_enc_width_mult + (depth_ratio - 1.0), # subtraction by 1 for max==min case
                                    num=args.delight_enc_layers,
                                    dtype=np.float
                                    )
        self.layers.extend(
            [DeLighTTransformerEncoderLayer(args=args,
                                            embed_dim=emb_dim,
                                            width_multiplier=round(width_multipliers[idx], 3),
                                            dextra_depth=layer_i)
                for idx, layer_i in enumerate(dextra_depths)
                ]
        )
        self.padding_idx = 0

    def forward(self, x):
        # compute padding mask
        encoder_padding_mask = x.eq(self.padding_idx).byte()

        x = self.emb(x)
        # pos = torch.cat([get_pos(x.shape[1]) for _ in range(x.shape[0])], dim=0)
        # pos 
        # print(x.shape, encoder_padding_mask.shape, self.pos_emb[:x.shape[1], :].shape)
        x += self.pos_emb[:x.shape[1], :] 
        # x = self.positional_dropout(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        x = x.mean(dim=1)
        x = x @ self.emb.weight.t()
        return x
