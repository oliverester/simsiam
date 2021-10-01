
class LabelHandler():
    """
    Collects and stores labels and their decoded pytorch mapping. Can be used to decode label to original label.
    """
    
    def __init__(self) -> None:
        
        self.encode_map = dict()
        self.decode_map = dict()
        
    def add_label(self, org_label):
        """Add label to labelhandler and assign encoded label - if not yet exists
        
        Args:
            org_label ([type]): [description]
        """
        if org_label not in self.encode_map.keys():
            next_encoded_label = len(self.encode_map)
            self.encode_map[org_label] = next_encoded_label
            self.decode_map[next_encoded_label] = org_label
        
    def encode(self, org_label):
        
        if org_label not in self.encode_map.keys():
            print(f"Adding new label '{org_label}' to LabelHandler")
            self.add_label(org_label)
            
        return self.encode_map[org_label]
    
    def decode(self, label):
        return self.decode_map[label]