import torch
import torch.nn as nn
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNN(nn.Module):
    #  TODO:  Implémenter comme décrit dans la question 1
    def __init__(self,input_dim,latent_dim,output_dim):

        super(RNN,self).__init__()
        self.input2hidden=nn.Linear(input_dim+latent_dim,latent_dim)
        self.activation1 = nn.Tanh()
        self.hidden2output=nn.Linear(latent_dim,output_dim)
        self.activation2=nn.Softmax()


    def one_step(self,x_t,h):
        combined=torch.cat((x_t,h),dim=-1)
        next_hidden=self.activation1(self.input2hidden(combined))
        return next_hidden

    def forward(self,x,h):
        length = x.size(0)
        batch_size = x.size(1)
        hidden_states = torch.zeros(length,batch_size, h.size(-1)).to(x.device)



        for t in range(length):
            x_t=x[t]
            h=self.one_step(x_t,h)
            hidden_states[t]=h

        return hidden_states


    def decode(self,h):

        output=self.hidden2output(h)
        output=self.activation2(output)

        return output

# Q3
class RNN2(nn.Module):

    def __init__(self, input_dim, latent_dim, output_dim, num_stations):
        super(RNN2, self).__init__()
        self.input2hidden = nn.Linear(input_dim * num_stations + latent_dim, latent_dim)
        self.activation1 = nn.Tanh()
        self.hidden2output = nn.Linear(latent_dim, output_dim )
        self.activation2 = nn.Sigmoid()
        self.num_stations = num_stations

    def one_step(self, x_t, h):
        combined = torch.cat((x_t, h), dim=-1)
        next_hidden = self.activation1(self.input2hidden(combined))
        return next_hidden

    def forward(self, x, h):
        length = x.size(0)
        batch_size = x.size(1)
        hidden_states = torch.zeros(length, batch_size, h.size(-1)).to(x.device)
        outputs = torch.zeros(length, batch_size, self.num_stations, self.hidden2output.out_features).to(x.device)

        for t in range(length):
            x_t = x[t]
            h = self.one_step(x_t, h)

            hidden_states[t] = h
            for station in range(self.num_stations):
                outputs[t, :, station, :] = self.decode(h)

        return outputs

    def decode(self, h):
        output = self.hidden2output(h)  # (batch_size,output_dim)
        output = self.activation2(output)
        return output

class RNN_trump(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim):
        super(RNN_trump,self).__init__()
        self.embedding=torch.nn.Embedding(vocab_size,embedding_dim)
        self.rnn=torch.nn.RNN(embedding_dim,hidden_dim,batch_first=True)
        self.fc=torch.nn.Linear(hidden_dim,output_dim)
        self.softmax=torch.nn.Softmax(dim=-1)
        self.hidden_dim=hidden_dim

    def forward(self,x,h):
        embedded=self.embedding(x)
        output,hidden=self.rnn(embedded,h)
        output=self.fc(output)
        output=self.softmax(output)
        return output,hidden
    def init_hidden(self,batch_size):
        return torch.zeros(1,batch_size,self.hidden_dim)




class SampleMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):

        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data, length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.classes*self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## transformation de l'index 1d vers une indexation 3d
        ## renvoie une séquence de longueur length et l'id de la station.
        station = i // ((self.nb_timeslots-self.length) * self.nb_days)
        i = i % ((self.nb_timeslots-self.length) * self.nb_days)
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length),station],station

class ForecastMetroDataset(Dataset):
    def __init__(self, data,length=20,horizon=2,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length,self.horizon= data,length,horizon
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.nb_days*(self.nb_timeslots - self.length-self.horizon+1)

    def __getitem__(self,i):
        ## Transformation de l'indexation 1d vers indexation 2d
        ## renvoie x[d,t:t+length-1,:,:], x[d,t+1:t+length,:,:]
        timeslot = i // self.nb_days
        day= i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length),:,:],self.data[day,(timeslot+self.horizon):(timeslot+self.length+self.horizon),:,:]

