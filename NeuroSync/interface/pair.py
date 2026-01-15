"""
Communication pair module for NeuroSync.
"""

from typing import Optional, Tuple, List
import torch

from NeuroSync.core.networks import Alice, Bob
from NeuroSync.crypto.keys import KeyManager
from NeuroSync.interface.cipher import NeuroSync
from NeuroSync.interface.sender import Sender
from NeuroSync.interface.receiver import Receiver


class CommunicationPair:
    """
    Synchronized communication pair.
    
    Creates matching sender and receiver that share
    the same key and network weights.
    """

    def __init__(self, cipher: NeuroSync):
        self.cipher = cipher
        self.sender = cipher.create_sender()
        self.receiver = cipher.create_receiver()
    
    @classmethod
    def from_pretrained(cls, dirpath: str) -> "CommunicationPair":
        """Loads a CommunicationPair from pretrained weights."""
        cipher = NeuroSync.from_pretrained(dirpath)
        return cls(cipher)
    
    def roundtrip(self, message: str) -> str:
        """
        Sends message and receives it (for testing).
        
        Args:
            message: Message to send
        
        Returns:
            Received message
        """

        packets = self.sender.send(message)
        
        result = None
        for packet in packets:
            received = self.receiver.receive(packet)
            if received is not None:
                result = received
        
        if result is not None:
            return result
        
        return ""
    
    def send(self, message: str) -> List:
        """Sends a message and returns the packets."""
        return self.sender.send(message)
    
    def receive(self, packet) -> Optional[str]:
        """Receives a packet and returns the message if complete."""
        return self.receiver.receive(packet)
    
    def reset(self) -> None:
        """Resets the sender and receiver state."""
        self.sender.sequence_counter = 0
        self.receiver.reset()
