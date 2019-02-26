import torch
import model
import utils
from dataloader.DiLiGenT_main import DiLiGenT_main 
from argParser import optionsTest
import torchvision as tv

args = optionsTest().parse()
torch.manual_seed(args.seed)

print('Instantiating test model')
in_c = 3 + args.in_light*3
model = model.PS_FCN_run(args.use_BN, in_c)

print('Loading Saved Model')
saved_model = torch.load(args.model_path)
if args.cuda:
    saved_model = saved_model.cuda()
    model = model.cuda()
model.load_state_dict(saved_model.state_dict())

test_set  = DiLiGenT_main(args, 'test')
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch,
                        num_workers=args.workers, pin_memory=args.cuda, shuffle=False)

model.eval()
print('---- Testing for %d images - DiLiGent Dataset ----' % (len(test_loader)))

err_mean = 0
with torch.no_grad():
    for i, sample in enumerate(test_loader):
        data = utils.parseData(args, sample, 'test')
        input = [data['input']]
        if args.in_light:
        	input.append(data['l'])
        output = model(input); 
        acc = utils.errorPred(data['tar'].data, output.data, data['m'].data) 
        err_mean = err_mean + acc
        print('error: %.3f' %(acc))
        result = (output.data + 1) / 2
        result_masked = result * data['m'].data.expand_as(output.data)
        
        save_path = './Results/' + 'img8_mask_%d.png' % (i+1)
        tv.utils.save_image(result_masked, save_path)
        print('saved image %d' %(i + 1))

print('------------ mean error: %.3f ------------' % (err_mean/len(test_loader))) 

