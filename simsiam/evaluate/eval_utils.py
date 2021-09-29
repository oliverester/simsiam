import torch 

def get_embeddings(val_loader, model, args):
   
    embeddings = torch.zeros(size=(len(val_loader.dataset), args.dim), dtype=torch.float32, device=args.device)
    labels = torch.zeros(size=(len(val_loader.dataset),), dtype=torch.int, device=args.device)
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pointer = 0
        for _, (images, targets) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            # compute output
            output = model(images)
            embeddings[pointer  : pointer+images.size(0)] = output
            labels[pointer : pointer+images.size(0)] = targets
            pointer += images.size(0)

    return embeddings.cpu().numpy(), labels.cpu().numpy()