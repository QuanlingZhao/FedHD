import logging
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

try:
    from fedml_core.distributed.client.client_manager import ClientManager
    from fedml_core.distributed.communication.message import Message
except ImportError:
    from FedML.fedml_core.distributed.client.client_manager import ClientManager
    from FedML.fedml_core.distributed.communication.message import Message
from .message_define import MyMessage
from .utils import transform_list_to_tensor
from .utils import transform_tensor_to_list


class FedHDClientManager(ClientManager):
    def __init__(self, mqtt_port, mqtt_host, args, trainer, comm=None, rank=0, size=0, backend="MQTT"):
        super().__init__(args, comm, rank, size, backend, mqtt_host, mqtt_port)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                              self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                              self.handle_message_receive_model_from_server)


    def handle_message_init(self, msg_params):
        global_hyper_vecs = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        
        #print("===========init s->c==================")
        #print(global_hyper_vecs)
        #print(type(global_hyper_vecs))
        #print("======================================")

        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        vecs_tensor = transform_list_to_tensor(gloal_hyper_ves)

        self.trainer.update_model(vecs_tensor)

        self.round_idx = 0
        self.__train()

    
    def start_training(self):
        self.round_idx = 0
        self.__train()
    

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        global_hyper_vecs = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        
        #print("===========update s->c==================")
        #print(global_hyper_vecs)
        #print(type(global_hyper_vecs))
        #print("======================================")

        vecs_tensor = transform_list_to_tensor(global_hyper_vecs)

        self.trainer.update_model(vecs_tensor)


        self.round_idx += 1
        self.__train()
        if self.round_idx == self.num_rounds - 1:
            self.finish()

    def send_model_to_server(self, receive_id, hyper_vecs, local_sample_num):
        #print("===========loacl_params c->s==================")
        #print(hyper_vecs)
        #print(type(hyper_vecs))
        #print("======================================")
        
        vecs_list = transform_tensor_to_list(hyper_vecs)
        
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, vecs_list)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)

    def __train(self):
        logging.info("#######training########### round_id = %d" % self.round_idx)
        hyper_vecs, local_sample_num = self.trainer.train()
        self.send_model_to_server(0, hyper_vecs, local_sample_num)
