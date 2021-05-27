import socket
from struct import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import pickle
import random



EMG8x_ADDRESS = '192.168.137.46'
CHANNELS_TO_MONITOR = (1,)

AD1299_NUM_CH = 8
TRANSPORT_BLOCK_HEADER_SIZE = 16
PKT_COUNT_OFFSET = 2
SAMPLES_PER_TRANSPORT_BLOCK = 64
TRANSPORT_QUE_SIZE = 4
TCP_SERVER_PORT = 3000
SPS = 1000
SAMPLES_TO_COLLECT = SAMPLES_PER_TRANSPORT_BLOCK * 8 * 45

TCP_PACKET_SIZE = int(((TRANSPORT_BLOCK_HEADER_SIZE) / 4 + (AD1299_NUM_CH + 1) * (SAMPLES_PER_TRANSPORT_BLOCK)) * 4)

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the port where the server is listening
server_address = (EMG8x_ADDRESS, TCP_SERVER_PORT)

sock.connect(server_address)

receivedBuffer = bytes()
rec_data = bytes()
rawSamples = np.zeros((SAMPLES_TO_COLLECT, len(CHANNELS_TO_MONITOR)))
# Collected samples
numSamples = 0
is_predicated = False
prd_array = np.array([])
brd_array = np.array([])
num_of_predications = 0
borders = []
classes = []
lb = pickle.loads(open("lb.pickle", "rb").read())
try:
    while True:
        if numSamples == (SAMPLES_TO_COLLECT - SAMPLES_PER_TRANSPORT_BLOCK) and not is_predicated:
            is_predicated = True
            num_of_predications = sock.recv(3)
            print("There are " + str(num_of_predications)[2] + str(num_of_predications)[3] + " predications")
            num_of_predications = int(str(num_of_predications)[2] + str(num_of_predications)[3])
            if num_of_predications == 0:
                break
            count = 0
            while True:
                prd_str = sock.recv(12)
                prd = sock.recv(4*3)
                brd = sock.recv(4*2)
                rec_data += prd + brd
                prd = np.frombuffer(prd, dtype=np.float32)
                brd = np.frombuffer(brd, dtype=np.int32)
                predication = np.array([prd])
                cl = lb.classes_[predication.argmax(axis=1)[0]]
                print(lb.classes_[predication.argmax(axis=1)[0]])
                classes.append(cl)
                borders.append(brd)
                # prd_array += prd
                # brd_array += brd
                print(prd)
                print(brd)
                count = count + 1
                if count == num_of_predications:
                    break
        if len(receivedBuffer) >= TCP_PACKET_SIZE * 2:
            # find sync bytes
            startOfBlock = receivedBuffer.find('EMG8x'.encode())

            if startOfBlock >= 0:

                # SAMPLES_PER_TRANSPORT_BLOCK*(AD1299_NUM_CH+1)+TRANSPORT_BLOCK_HEADER_SIZE/4
                strFormat = '{:d}i'.format(
                    round(SAMPLES_PER_TRANSPORT_BLOCK * (AD1299_NUM_CH + 1) + TRANSPORT_BLOCK_HEADER_SIZE / 4))
                # '1156i'
                samples = unpack(strFormat, receivedBuffer[startOfBlock:startOfBlock + TCP_PACKET_SIZE])

                # remove block from received buffer
                receivedBuffer = receivedBuffer[startOfBlock + TCP_PACKET_SIZE:]

                chCount = 0
                for chIdx in CHANNELS_TO_MONITOR:
                    # get channel offset
                    offset_toch = int(TRANSPORT_BLOCK_HEADER_SIZE / 4 + SAMPLES_PER_TRANSPORT_BLOCK * chIdx)

                    # print( samples[offset_to4ch:offset_to4ch+SAMPLES_PER_TRANSPORT_BLOCK] )
                    dataSamples = samples[offset_toch:offset_toch + SAMPLES_PER_TRANSPORT_BLOCK]

                    blockSamples = np.array(dataSamples)
                    #print(len(blockSamples))
                    print('Ch#{0} Block #{1} mean:{2:10.1f},  var:{3:8.1f}, sec:{4:4.0f}'.format(chIdx, samples[
                        PKT_COUNT_OFFSET], np.mean(blockSamples), np.var(blockSamples) / 1e6, numSamples / SPS))

                    rawSamples[numSamples:numSamples + SAMPLES_PER_TRANSPORT_BLOCK, chCount] = blockSamples

                    chCount += 1

                numSamples += SAMPLES_PER_TRANSPORT_BLOCK
                if numSamples >= SAMPLES_TO_COLLECT:
                    break

        else:
            receivedData = sock.recv(TCP_PACKET_SIZE)
            if not receivedData:
                print("а теперь тут")
                # probably connection closed
                break

            receivedBuffer += receivedData


finally:
    sock.close()
x = rawSamples
x     -= np.mean(x)
SPS = 1000.0
x = np.array(x)
x = np.reshape(x, len(x))
print(x)
hflt        = signal.firls( 513, [ 0.,5., 7.,SPS/2 ], [     0.,0., 1.0,1.0  ], fs = SPS)
y = np.convolve( hflt, x, 'same' )




# for i in range(num_of_predications):
#     print(lb.classes_[prd_array[i].argmax(axis=1)[0]])
fig, ax = plt.subplots()
#ax.figure(figsize=(20,6))
plt.plot(y)
for i in range(num_of_predications):
    print(borders[i][0], borders[i][1])
    if int(classes[i]) == 1:
        ax.vlines(borders[i][0], -300000, 300000, color='red')
        ax.vlines(borders[i][1], -300000, 300000, color='red')
        ax.hlines(-300000, borders[i][0], borders[i][1], color='red')
    if int(classes[i]) == 2:
        ax.vlines(borders[i][0], -300000, 300000, color='yellow')
        ax.vlines(borders[i][1], -300000, 300000, color='yellow')
        ax.hlines(-300000, borders[i][0], borders[i][1], color='yellow')
    if int(classes[i]) == 3:
        ax.vlines(borders[i][0], -300000, 300000, color='green')
        ax.vlines(borders[i][1], -300000, 300000, color='green')
        ax.hlines(-300000, borders[i][0], borders[i][1], color='green')

plt.grid(True)
plt.show()
