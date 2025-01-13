import torch
import torch.nn as nn
import torchvision.models as models
import statistics

class EncoderCNN(nn.Module):
    def __init__(self,embed_size,train_CNN=False):
        super().__init__
        self.train_CNN = train_CNN
        self.vgg = models.vgg16(pretrained=True)
        self.vgg.classifier = nn.Sequential(nn.Linear(self.vgg.classifier[6].in_features, embed_size),nn.ReLU(),nn.Dropout(0.5))
        for param in self.vgg.features.parameters():
            param.requires_grad = False
        for param in self.vgg.classifier.parameters():
            param.requires_grad = True
    def forward(self,images):
        features = self.vgg(images)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super().__init__
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.dropout(0.5)

    def forward(self,features,captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim = 0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super().__init__
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
    def forward(self, images, captions):
        features = self.encoderCNN(images)        
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_images(self, image, vocabulary, max_size = 50):
        result_caption = []
        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None
            for _ in range(max_size):
                hiddens, states = self.decoderRNN.lstm(x,states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == '<EOS>':
                    break
        return [vocabulary.itos[idx] for idx in result_caption]

# model = models.vgg16(pretrained=True)
# model.classifier = nn.Sequential(nn.Linear(model.classifier[0].in_features, 10),nn.ReLU(),nn.Dropout(0.5))
# for name, param in model.classifier.named_parameters():
#     print(f"Parameter name: {name}")
#     print(f"Parameter value:\n{param.data}\n")