from socket import *
from struct import pack
import random
import string
from time import sleep, monotonic
import argparse
import os


class ClientProtocol:

    def __init__(self):
        self.socket = None
        self.output_dir = '.'

    def parse_arguments(self):
        parser = argparse.ArgumentParser(
            description='Measure latency using TCP.',
        )
        parser.add_argument(
            '--host',
            default='155.98.38.150',
        )
        parser.add_argument(
            '--port',
            default=8888,
            type=int,
        )
        parser.add_argument(
            '--time',
            default=1000,
            type=int,
        )
        parser.add_argument(
            '--size',
            default=5632,
            type=int,
        )
        return parser.parse_args()

    def connect(self, server_ip, server_port):
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.connect((server_ip, server_port))

    def close(self):
        self.socket.shutdown(SHUT_WR)
        self.socket.close()
        self.socket = None

    def send_data(self, data, size):

        # use struct to make sure we have a consistent endianness on the length
        length = pack('>Q', len(data))

        s_start = monotonic()
        # sendall to make sure it blocks if there's back-pressure on the socket
        self.socket.sendall(length)
        self.socket.sendall(data)

        ack = self.socket.recv(1)
        s_runtime = (monotonic() - s_start) * 1000
        bandwidth = args.size*1024 / (s_runtime/1000)
        # print(s_runtime)
        # print(bandwidth)
        # print("---------")

        with open(os.path.join(self.output_dir, '%06d' % size), 'a+') as fp:
            fp.write(f'{s_runtime} {bandwidth}\n')

        # could handle a bad ack here, but we'll assume it's fine.

if __name__ == '__main__':
    cp = ClientProtocol()
    args = cp.parse_arguments()
    # image_data = None
    # with open('IMG_0077.jpg', 'r') as fp:
    #     image_data = fp.read()
    data = str.encode(''.join(random.choices(string.ascii_uppercase + string.digits, k=args.size*1024)))
    print(len(data))
    assert(len(data))
    for i in range(args.time):
        cp.connect(args.host, args.port)
        cp.send_data(data, args.size)
        cp.close()
