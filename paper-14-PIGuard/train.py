import os
import torch
import numpy as np
from transformers import AutoTokenizer, get_scheduler
from torch.utils.data import DataLoader
from PIGuard import PIGuard
from datasets import load_dataset
from util import set_seed, get_logger, compute_accuracy
from params import parse_args
from eval import evaluate


def train():
    global logger
    args = parse_args()

    set_seed(args)
    logger = get_logger(os.path.join(args.logs, "log_{}.txt".format(args.name)))

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # init train setting
    epochs = args.epochs
    save_path = args.checkpoint_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Directory '{save_path}' created.")
    else:
        print(f"Directory '{save_path}' already exists.")

    # tokenizer initial
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')

    def preprocess_function(examples):
        encoding_text = tokenizer(examples['prompt'], padding='max_length', truncation=True, max_length=args.max_length)
        
        return {
            'input_ids': encoding_text['input_ids'],                 
            'attention_mask': encoding_text['attention_mask'],        
        }

    # Prepare dataset
    data_files = {
        "train": args.train_set,
        "valid": args.valid_set,
    }

    dataset = load_dataset('json', data_files=data_files)

    encoded_dataset = dataset.map(preprocess_function, batched=True)
    encoded_dataset = encoded_dataset.map(lambda examples: {'labels': [label for label in examples['label']]}, batched=True)
    encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    train_loader = DataLoader(encoded_dataset['train'], batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(encoded_dataset['valid'], batch_size=args.eval_batch_size, shuffle=False)

    model = PIGuard('microsoft/deberta-v3-base', num_labels=2, device=device) 

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device), strict=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps)

    scheduler = get_scheduler(
        name="linear",                
        optimizer=optimizer,          
        num_warmup_steps=args.warmup, 
        num_training_steps=epochs * len(train_loader) 
    )

    best_accuracy = 0
    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

            optimizer.zero_grad()
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            logits, loss = model(input_ids, attention_mask, labels, mode="train")

            loss.backward()

            optimizer.step()
            scheduler.step()

            if step % args.display == 0:
                logger.info(f"Step: {step} / {len(train_loader)}.")
                logger.info(f"Loss: {loss:.3f}")

            if ((step % args.save_step == 0) and (step != 0)) or (step == (len(train_loader) - 1)):
                model.eval()
                loss_list, logits_list, labels_list = [], [], []

                with torch.no_grad():
                    for eval_step, batch in enumerate(validation_loader):
                        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
                        optimizer.zero_grad()
                        input_ids = input_ids.to(device)
                        attention_mask = attention_mask.to(device)
                        labels = labels.to(device)
                        
                        logits, loss = model(input_ids, attention_mask, labels)

                        loss_list.append(loss.cpu().item())
                        logits_list.append(logits.cpu())        
                        labels_list.append(labels.cpu())

                        if eval_step % args.display == 0:
                            logger.info(f"Step: {eval_step} / {len(validation_loader)}.")
                            logger.info(f"Loss: {loss:.3f}")

                    combined_logits = torch.cat(logits_list, dim=0)
                    combined_labels = torch.cat(labels_list, dim=0)

                    pred = combined_logits.argmax(1)

                    benign_accuracy, injection_accuracy, total_accuracy = compute_accuracy(pred, combined_labels)

                    print(f"total accuracy: {total_accuracy}")
                    print(f"benign accuracy: {benign_accuracy}")
                    print(f"injection accuracy: {injection_accuracy}")
                    print(f"loss: {np.mean(loss_list)}")

                    # eval on valid set
                    if total_accuracy > args.save_thres:
                        model_path = f'{save_path}/epoch_{epoch}_{step}_model.pth'
                        torch.save(model.state_dict(), model_path)
                        print(f"Saved to {model_path}.")

                        if total_accuracy > best_accuracy:
                            best_accuracy = total_accuracy
                            best_model_path = f'{save_path}/best_model.pth'
                            torch.save(model.state_dict(), best_model_path)
                            print(f"Saved to {best_model_path}.")

                model.train()

    # evaluate on overall test set
    logger.info("Evaluate Best Model on Test Sets.")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    logger.info(f"Loaded model from {best_model_path}.")
    evaluate(model, args.dataset_root)

if __name__ == "__main__":
    train()