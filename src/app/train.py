import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from data.dataset import *
from utils import *


class Trainer():
    def __init__(self, model, data, args):

        self.args = args
        self.model = model.to(args.device)

        self.train_loader = data['train']
        self.val_loader = data['val']
        self.log = Log(args)
        self.device = args.device
        if args.checkpoint:
            self._load_checkpoint(args.model_path)
        
        if args.freeze:
            self._frzee_model_parameters()

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                          lr=args.learning_rate)
        
        self.scheduler = ReduceLROnPlateau(self.optimizer, 
                                           mode='min',
                                           factor=0.5,
                                           patience=2,
                                           verbose=True)
    def _frzee_model_parameters(self):
        for p_name, param in self.model.named_parameters():
            if "MFB" in p_name and "radar_net" not in p_name:
                param.requires_grad = False
                print("freeze:", p_name)

    def _load_checkpoint(self, model_path):
        print("load checkpoint")
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['state_dict'])

    def train(self):
        
        best_train_loss = 1e5
        best_val_loss = 1e5
        for epoch in range(self.args.epoch):
            torch.cuda.empty_cache()
            self.model.train()
            train_loss = self.train_epoch(self.train_loader, epoch=epoch)
            checkpoint = {'loss': train_loss,
                          'state_dict': self.model.state_dict(),
                          'optimizer': self.optimizer.state_dict()}
            self.log.save(checkpoint, "temp_train.pth")

            if best_train_loss > train_loss:
                best_train_loss = train_loss
                self.log.save(checkpoint, "best_train.pth")

            if epoch % self.args.val_per_epoch == 0:
                self.model.eval()

                with torch.no_grad():
                    val_loss, val_snr = self.test_epoch(self.val_loader,
                                                        epoch=epoch)

                self.log.add_scalar(cate="val",
                                    global_step = epoch//self.args.val_per_epoch,
                                    val_loss = val_loss,
                                    val_snr = val_snr)
                
                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    self.log.save(checkpoint, "best_val.pth")    
                print("epoch: {:03d} | trn loss: {:.8f} |val loss: {:.8f}"
                      .format(epoch, train_loss ,val_loss))               
    
    def train_batch(self, batch_data):
        raise Exception("should never be called")
    
    def test_batch(self, batch_data):
        raise Exception("should never be called")

    def train_epoch(self, data_loader, epoch):
        total_loss = 0

        for i, batch_data in enumerate(tqdm(data_loader)):
            loss, _ = self.run_batch(batch_data)
            total_loss += loss["loss"].item()
            
            self.optimizer.zero_grad()
            loss["loss"].backward()
            original_grad = nn.utils.clip_grad_norm_(self.model.parameters(),
                                                     max_norm=5,
                                                     norm_type=2)
            self.optimizer.step()
            self.log.add_scalar(cate="train",
                                global_step = epoch*len(data_loader) + i,
                                **loss
                                )
            self.log.add_scalar(cate="train",
                                global_step = epoch*len(data_loader) + i,
                                original_grad = original_grad
                                )
        self.scheduler.step(total_loss/(i+1))
        return total_loss/(i+1)
    
    def test_epoch(self, data_loader, epoch):
        total_loss = 0
        total_snr = 0
        for i, batch_data in enumerate(tqdm(data_loader)):
            loss, est_audio = self.run_batch(batch_data)
            total_loss += loss["loss"].item()
        clean_audio = batch_data["clean"][0]
        est_audio = est_audio.squeeze(0)
        est_audio_dict = {f"speaker{i}":est_audio[i] for i in range(clean_audio.shape[0])}
        clean_audio_dict = {f"speaker{i}":clean_audio[i] for i in range(clean_audio.shape[0])}

        self.log.add_audio(cate="test/mix",
                           global_step = epoch,
                           add_spec=True,
                           noisy=batch_data["mix"][0],
                           )
        self.log.add_audio(cate="test/est",
                           global_step = epoch,
                           add_spec=True,
                           **est_audio_dict,
                           )
        self.log.add_audio(cate="test/clean",
                           global_step = epoch,
                           add_spec=True,
                           **clean_audio_dict,
                           )

        return total_loss/(i+1), total_snr/(i+1)