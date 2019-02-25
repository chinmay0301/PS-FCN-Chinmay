from argParser import optionsTrain
import torch
import model
import utils
from dataloader.PS_Synth_Dataset import PS_Synth_Dataset

args = optionsTrain().parse()

print('Model Instatiation')

in_c = 3 + args.in_light*3
model = model.PS_FCN(args.use_BN, in_c,)

if args.cuda: 
    model = model.cuda()

print(" Loading Data from %s" % (args.data_dir))

train_set = PS_Synth_Dataset(args, args.data_dir, 'train')
print('Training set loaded')

train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch,
                    num_workers=args.workers, pin_memory=args.cuda, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), args.init_lr, betas=(args.beta_1, args.beta_2))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, 
                gamma=args.lr_decay, last_epoch=-1)
if args.masked_loss:
    print('using mask')
    criterion = utils.Criterion_mask(args)
else:
    criterion = utils.Criterion(args)
model.train()
for epoch in range(1, args.epochs + 1):
    print('---- Start Training Epoch %d: %d batches ----' % (epoch, len(train_loader)))
    scheduler.step()
    for i, sample in enumerate(train_loader):
        data  = utils.parseData(args, sample, 'train')
        input = [data['input']];
        if args.in_light: 
            input.append(data['l'])
        output = model(input) 
        optimizer.zero_grad()
        loss = criterion.forward(output, data['tar']); 
        criterion.backward() 
        optimizer.step() 
    print("Loss in epoch %d: %.3f" %(epoch, loss))

torch.save(model, './TrainedModels/model_new.pth.tar')
print("saved the model")
