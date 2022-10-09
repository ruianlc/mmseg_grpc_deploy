from __future__ import print_function

import os
import argparse
import grpc

# import dl_b64_pb2
# import dl_b64_pb2_grpc
import dl_abi_pb2
import dl_abi_pb2_grpc

from util.img_util import convert_base64_to_PIL, save_image


def get_vs_rootpath():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return dir_path

def run(host, port):
    channel = grpc.insecure_channel('%s:%d' % (host, port))
    stub = dl_abi_pb2_grpc.PredictorStub(channel)
    request = dl_abi_pb2.PredictRequest(
        model_path=os.path.join(get_vs_rootpath(), 'model/swin_tobacco_30000.pth'),
        config_path=os.path.join(get_vs_rootpath(), 'model-config/swin_tobacco.py'),
        img_path=os.path.join(get_vs_rootpath(), 'data/input/test/26.png'),
        redun_id='1101',
    )
    response = stub.predict(request)

    output_file = os.path.join(get_vs_rootpath(), 'data/output/img_seg.png')
    save_image(convert_base64_to_PIL(response.image), output_file)
    print(f'the inference results:\n{response.res_g_ratio}, {response.res_y_ratio}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', help='host name', default='localhost', type=str)
    parser.add_argument('--port', help='port number', default=50052, type=int)

    args = parser.parse_args()
    run(args.host, args.port)
