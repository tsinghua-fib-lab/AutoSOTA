import torch
import numpy as np

all_tasks = ["assoc_recall", "assoc_recall_mk", "binary_copy", "binary_encode", "binary_recall", "binary_recall_mix", "binary_recall_last",\
            "binary_recall_rep", "median_last", "parity", "read_write", "sparse_parity", "threshold", "var_copy"]

def set_task_specific_parameters(args):
    assert args.task_name in all_tasks, "Task not in list of all implemented tasks"
    
    if args.task_name == "assoc_recall":
        args.num_numbers = 0
        args.vocab_size = args.num_vocab + 1
        args.data_name = "data_%d_%d" % (args.sequence_len, args.num_vocab)

    if args.task_name == "assoc_recall_mk":
        print("Shifting the number of vocab to hits are as frequent")
        args.size_query = 2
        args.num_vocab = int(1 + args.num_vocab ** (1./args.size_query))
        args.vocab_size = args.num_vocab + 1
        args.data_name = "data_%d_%d" % (args.sequence_len, args.num_vocab)
        print("Current query length:", args.size_query)

    if args.task_name == "binary_copy":
        args.vocab_size = 2 + args.num_vocab + 1
        args.data_name = "data_%d_%d_%d" % (args.sequence_len, args.num_vocab)
        print("Using %d bits" % args.num_bits)

    if args.task_name == "binary_encode":
        args.vocab_size = 2 ** args.num_bits + 1
        args.data_name = "data_%d_%d" % (args.sequence_len, args.num_bits)
        print("Using %d bits" % args.num_bits)

    if args.task_name == "binary_recall":
        args.vocab_size = 2 + 2 ** args.num_bits + 1
        args.data_name = "data_%d_%d" % (args.sequence_len, args.num_bits)
        print("Using %d bits" % args.num_bits)

    if args.task_name == "binary_recall_mix":
        args.vocab_size = 2 + 2 ** args.num_bits + 1
        args.data_name = "data_%d_%d" % (args.sequence_len, args.num_bits)
        print("Using %d bits" % args.num_bits)

    if args.task_name == "binary_recall_last":
        args.vocab_size = 2 + 2 ** args.num_bits + 1
        args.data_name = "data_%d_%d.pt" % (args.sequence_len, args.num_bits)
        print("Using %d bits" % args.num_bits)

    if args.task_name == "binary_recall_rep":
        args.vocab_size = 2 + 2 ** args.num_bits + 1
        args.data_name = "data_%d_%d.pt" % (args.sequence_len, args.num_bits)
        print("Using %d bits" % args.num_bits)

    if args.task_name == "median_last":
        assert args.num_numbers % 2 == 1
        
        args.vocab_size = args.num_numbers + 1
        args.data_name = "data_%d_%d" % (args.sequence_len, args.num_numbers)

    if args.task_name == "parity":
        args.vocab_size = 2 + 1
        args.data_name = "data_%d" % (args.sequence_len,)

    if args.task_name == "read_write":
        args.num_numbers = 0
        args.vocab_size = args.num_vocab + 3
        args.data_name = "data_%d_%d" % (args.sequence_len, args.num_vocab)

    if args.task_name == "sparse_parity":
        args.vocab_size = 2 + 1
        args.data_name = "data_%d" % (args.sequence_len,)

    if args.task_name == "threshold":
        args.vocab_size = 2 + args.num_numbers + args.num_vocab + 1
        args.data_name = "data_%d_%d_%d" % (args.sequence_len, args.num_numbers, args.num_vocab)

    if args.task_name == "var_copy":
        args.min_num_token = 5
        
        args.vocab_size = args.min_num_token + args.num_numbers + args.num_vocab + 1
        args.data_name = "data_%d_%d_%d" % (args.sequence_len, args.num_numbers, args.num_vocab)



def generate_seq(args, ood=False):
    if args.task_name == "assoc_recall":
        seq_in = torch.randint(0, args.num_vocab, (args.sequence_len,))
        seq_out = torch.zeros(args.sequence_len, dtype=torch.int32) - 1 # All empty tokens
    
        lookup = {}
        for i in range(1, args.sequence_len):
            lookup[seq_in[i-1].item()] = seq_in[i].item()
            if seq_in[i].item() in lookup.keys():
                seq_out[i] = lookup[seq_in[i].item()]
    
    if args.task_name == "assoc_recall_mk":
        seq_in = torch.randint(0, args.num_vocab, (args.sequence_len,))
        seq_out = torch.zeros(args.sequence_len, dtype=torch.int32) - 1 # All empty tokens
    
        lookup = {}
        for i in range(args.size_query, args.sequence_len):
            key = tuple(seq_in[i-args.size_query:i].tolist())
            lookup[key] = seq_in[i].item()

            key = tuple(seq_in[i-args.size_query+1:i+1].tolist())
            if key in lookup.keys():
                seq_out[i] = lookup[key]
    
    if args.task_name == "binary_copy":
        seq_in = torch.randint(2, 2 + args.num_vocab, (args.sequence_len,))
        seq_in[torch.multinomial(torch.tensor([1-args.p, args.p]), args.sequence_len, replacement=True).to(torch.bool)] = 0
        seq_in[torch.multinomial(torch.tensor([1-args.p, args.p]), args.sequence_len, replacement=True).to(torch.bool)] = 1
    
        seq_out = torch.zeros(args.sequence_len, dtype=torch.int32) - 1
        s = 0
        for i in range(args.sequence_len):
            if seq_in[i] in [0, 1]:
                s = (2*s + seq_in[i]) % (2 ** args.num_bits)

            if i-s >= 0:
                seq_out[i] = seq_in[i-s]
                
    if args.task_name == "binary_encode":
        seq_in = torch.randint(0, 2, (args.sequence_len,))
        seq_out = torch.zeros(args.sequence_len, dtype=torch.int32)
        s = 0
        for i in range(args.sequence_len):
            s = (2*s + seq_in[i]) % (2 ** args.num_bits)
            seq_out[i] = s
    
    if args.task_name == "binary_recall":
        seq_in = torch.randint(2, 2 + 2 ** args.num_bits, (args.sequence_len,))
        
        seq_in[torch.multinomial(torch.tensor([1-args.p, args.p]), args.sequence_len, replacement=True).to(torch.bool)] = 0
        seq_in[torch.multinomial(torch.tensor([1-args.p, args.p]), args.sequence_len, replacement=True).to(torch.bool)] = 1 
    
        seq_out = torch.zeros(args.sequence_len, dtype=torch.int32) -1
        assoc = {i: -1 for i in range(2, 2 + 2**args.num_bits)}
        s = 0
        for i in range(args.length):
            if seq_in[i] in [0, 1]:
                s = (2*s + seq_in[i].item()) % (2**args.num_bits)

            if i >= 1 and 2 <= seq_in[i-1] and seq_in[i-1] < 2 + 2**args.num_bits: # Is a vocab token
                assoc[seq_in[i-1].item()] = seq_in[i]

            seq_out[i] = assoc[s+2]
    
    if args.task_name == "binary_recall_mix":
        target = torch.randint(0, 2 ** args.num_bits, (1,)).item()
        
        seq_in = torch.randint(2, 2 + (2 ** args.num_bits), (args.sequence_len,)) # Only 32 possible tokens, can be set to more
        temp = target

        bit_positions = np.random.randint(0, 2 ** args.num_bits - args.num_bits // 2, args.num_bits // 2).tolist()
        bit_positions += list(range(args.sequence_len-1, args.sequence_len-1-args.num_bits // 2, -1))
        bit_positions.sort()
        
        for i in bit_positions:
            seq_in[i] = temp % 2
            temp = temp // 2
    
        assoc = {i: 0 for i in range(2 + 2 ** args.num_bits)}
        for i in range(1,args.sequence_len-args.num_bits):
            assoc[seq_in[i-1].item()] = seq_in[i]
    
        seq_out = torch.zeros(args.sequence_len, dtype=torch.int32) -1
        seq_out[-1] = assoc[2 + target]
    
    if args.task_name == "binary_recall_last":
        target = torch.randint(0, 2 ** args.num_bits, (1,)).item()
        
        seq_in = torch.randint(2, 2 + (2 ** args.num_bits), (args.sequence_len,)) # Only 32 possible tokens, can be set to more
        temp = target
        for i in range(args.sequence_len-1, args.sequence_len-1-args.num_bits, -1):
            seq_in[i] = temp % 2
            temp = temp // 2
    
        assoc = {i: 0 for i in range(2 + 2 ** args.num_bits)}
        for i in range(1,args.sequence_len-args.num_bits):
            assoc[seq_in[i-1].item()] = seq_in[i]
    
        seq_out = torch.zeros(args.sequence_len, dtype=torch.int32) -1
        seq_out[-1] = assoc[2 + target]

    if args.task_name == "binary_recall_rep":
        target = torch.randint(0, 2 ** args.num_bits, (1,)).item()

        repeat_length = torch.randint(4, 10, (1,)).item()
        seq_in = torch.randint(2, 2 + (2 ** args.num_bits), (repeat_length,)) # Only 32 possible tokens, can be set to more
        seq_in = seq_in.repeat(int(args.sequence_len / repeat_length) + 1)[:args.sequence_len]
        
        temp = target
        for i in range(args.sequence_len-1, args.sequence_len-1-args.num_bits, -1):
            seq_in[i] = temp % 2
            temp = temp // 2
    
        assoc = {i: 0 for i in range(2 + 2 ** args.num_bits)}
        for i in range(1,args.sequence_len-args.num_bits):
            assoc[seq_in[i-1].item()] = seq_in[i]
    
        seq_out = torch.zeros(args.sequence_len, dtype=torch.int32) -1
        seq_out[-1] = assoc[2 + target]
    
    if args.task_name == "median_last":
        seq_in = torch.randint(0, args.num_numbers, (args.sequence_len,))
        seq_in[-1] = 1 # Marks that the models should be outputting a median
    
        seq_out = torch.zeros(args.sequence_len, dtype=torch.int32) -1
        seq_out[-1] = torch.median(seq_in[-args.num_numbers-1:-2])
    
    if args.task_name == "parity":
        seq_in = torch.multinomial(torch.Tensor([0.5, 0.5]), args.sequence_len, replacement=True)
        seq_in[0] = 0 # Distribution choice to make code simpler
        seq_out = torch.zeros(args.sequence_len, dtype=torch.int32)
    
        for i in range(1, args.sequence_len):
            seq_out[i] = (seq_out[i-1] + seq_in[i]) % 2

    if args.task_name == "read_write":
        # -3 is the read token
        # -2 is the write token
        # -1 is the ignore token
        seq_in = torch.randint(0, args.num_vocab, (args.sequence_len+1,))
        seq_in[::2] = torch.multinomial(torch.Tensor([0.1, 0.1, 0.8]), args.sequence_len // 2 + 1, replacement=True).to(torch.int32) - 3
        
        lookup = {}
        for i in range(0, args.sequence_len, 2):
            if seq_in[i].item() == -2: # Write token
                lookup[i] = seq_in[i+1].item()  
            elif seq_in[i].item() == -3: # Read token
                if i in lookup.keys():
                    seq_in[i+1] = lookup[i]
                # else leave it as is (random token)
        
        seq_out = seq_in[1:] # Shifted by one
        seq_in = seq_in[:-1]
    
    if args.task_name == "sparse_parity":
        seq_in = torch.multinomial(torch.Tensor([1-args.p, args.p]), args.sequence_len, replacement=True)
        seq_in[0] = 0 # Distribution choice to make code simpler
        seq_out = torch.zeros(args.sequence_len, dtype=torch.int32)
    
        for i in range(1, args.sequence_len):
            seq_out[i] = (seq_out[i-1] + seq_in[i]) % 2
    
    if args.task_name == "threshold":
        threshold = args.num_numbers / args.p # this just seems to work, nothing more special than that
        
        seq_in = torch.zeros(args.sequence_len, dtype=torch.int32)
        seq_out = torch.zeros(args.sequence_len, dtype=torch.int32) -1
    
        lookup = {}
        s = 0
        for i in range(args.length):
            if torch.rand((1,)) > args.p:
                # Add a random vocab token
                seq_in[i] = torch.randint(args.num_numbers+2, args.num_numbers+args.num_vocab+2, (1,))
            else:
                # Add a random number token
                num = torch.randint(0, args.num_numbers, (1,))
                seq_in[i] = num + 2
                s += num # ... which is added to the sum
    
            if s < threshold:
                seq_out[i] = 0
            else:
                seq_out[i] = 1
  
    if args.task_name == "var_copy":
        vocab = torch.randint(args.min_num_token+args.num_numbers, args.min_num_token+args.num_numbers+args.num_vocab, (args.sequence_len,), dtype=torch.int32)
        numbers = torch.randint(args.min_num_token, args.min_num_token+args.num_numbers, (args.sequence_len,), dtype=torch.int32)
        mask = torch.rand_like(vocab, dtype=torch.float) < args.p

        seq_in = torch.where(mask, numbers, vocab)
        seq_out = torch.zeros(args.sequence_len, dtype=torch.int32) -1

        number_seq = [(i, v) for i, v in enumerate(seq_in) if args.min_num_token <= v and v < args.min_num_token+args.num_numbers]
        for i in range(len(number_seq)-1):
            this_i = number_seq[i][0]
            this_v = number_seq[i][1]
            next_i = number_seq[i+1][0]
            next_v = number_seq[i+1][1]

            if next_i-this_v > 0:
                if this_i-this_v >= 0:
                    seq_out[this_i:next_i] = seq_in[this_i-this_v:next_i-this_v]
                else:
                    seq_out[this_v:next_i] = seq_in[:next_i-this_v]

        if len(number_seq) > 0:
            this_i = number_seq[-1][0]
            this_v = number_seq[-1][1]

            if this_i-this_v >= 0:
                seq_out[this_i:] = seq_in[this_i-this_v:-this_v]
            else:
                seq_out[this_v:] = seq_in[:-this_v]

    return seq_in, seq_out


def generate_data(args, all_at_once=True):
    sequences_in = []
    sequences_out = []

    if all_at_once:
        print("Generating Data")
        for i in range(args.batch_size * args.batches_per_epoch):
            if i % 8000 == 0:
                print("- Sequence Num:", i)
            seq_in, seq_out = generate_seq(args)
    
            sequences_in.append(seq_in)
            sequences_out.append(seq_out)
    else:
        for i in range(args.batch_size):
            seq_in, seq_out = generate_seq(args)
    
            sequences_in.append(seq_in)
            sequences_out.append(seq_out)

    x_in = torch.stack(sequences_in)  # (batch_size, max_len)
    x_out = torch.stack(sequences_out)  # (batch_size, max_len)

    # Shift -1
    x_in = (x_in + args.vocab_size) % args.vocab_size
    x_out = (x_out + args.vocab_size) % args.vocab_size

    return x_in, x_out