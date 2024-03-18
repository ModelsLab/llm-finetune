from datasets import *


class DataLoader:
    """
    Base dataset loader class which used to load dataset in finetuning env
    """

    def __init__(self,dataset_id : str,split : str = None) -> None:
        self.dataset_id = dataset_id
        self.dataset_builder = None
        self.split = split
        self.dataset = None
        

    def load(self):
        pass

    def check_dataset(self):
        self.dataset_builder = load_dataset_builder(self.dataset_id)
        info = self.get_dataset_info()
        ## TODO : ADD all the required fields for data validation like size and stuff
        if info.dataset_size > 2 * 1024 * 1024 * 1024:
            return False
        else :
            return True
        
        ## return true if data is in correct for and follows our requirements.
        return True
    
    def load_from_local_disk(self):
        pass

    
    def load_from_hf(self,split : str = None):
        if split :
            self.dataset = load_dataset(self.dataset_id,split=split)
        else :
            self.dataset = load_dataset(self.dataset_id)


    
    def get_dataset_info(self):
        if self.dataset_builder:
            return self.dataset_builder.info
        else :
            return None



obj = DataLoader("rotten_tomatoes")

print(obj.check_dataset())