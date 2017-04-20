import os
import torch
import librosa


class Corpus(object):

    def __init__(self, path, mfcc_dim=40, fps=50, bs_dim=46, cuda=False):
        self.mfcc_dim = mfcc_dim
        self.fps = fps
        self.bs_dim = bs_dim
        self.cuda = False

        self.audio = self.audio_feature_extraction(path)
        self.video = self.video_feature_extraction(path)

    def data(self):
        return list(zip(self.audio, self.video))

    def audio_feature_extraction(self, path):
        file_list = [os.path.join(path, f) for f in os.listdir(path) if
                     f[0] != '.']

        file_list = file_list[:100]

        corpus_feature = []
        for file in file_list:
            print('load file: {}'.format(file))
            y, sr = librosa.load(file)
            hop_length = sr // self.fps  # re-sampling to align with the video
            mfcc = librosa.feature.mfcc(y, sr, hop_length=hop_length,
                                        n_mfcc=self.mfcc_dim)
            # covert numpy to pytorch.Tensor
            data = torch.from_numpy(mfcc).t().contiguous()
            data = data.float()
            if self.cuda:
                data = data.cuda()
            corpus_feature += [data]

        return corpus_feature

    def video_feature_extraction(self, path):
        video_feature = []
        for mfcc in self.audio:
            bs_seq = torch.rand(mfcc.size(0), self.bs_dim)
            bs_seq = mfcc
            if self.cuda:
                bs_seq = bs_seq.cuda()
            video_feature += [bs_seq]

        return video_feature
