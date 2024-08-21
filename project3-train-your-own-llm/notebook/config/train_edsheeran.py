out_dir = "out-lyrics"
eval_interval = 250  # keep frequent because we'll overfit
eval_iters = 20
log_interval = 10  # don't print too often
# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False
dataset = "ed-sheeran"
batch_size = 12  # 12 samples per iteration
block_size = 64  # context size
# a baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384 # each embedding vector for each token will have 384 dimensions
dropout = 0.2
learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 2000
lr_decay_iters = 2000  # make equal to max_iters usually
min_lr = 1e-4  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small
warmup_iters = 100  # not super necessary potentially