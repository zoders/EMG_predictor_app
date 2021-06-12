import window
import sys
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
import socket
from struct import *
import numpy as np
from scipy import signal
import pickle
from PyQt5 import QtCore, QtWidgets
# https://pypi.org/project/qt-material/
from qt_material import apply_stylesheet


class GetThread(QtCore.QThread):
    signal = QtCore.pyqtSignal(str)
    rawSamples = np.zeros(0)
    EMG8x_ADDRESS = "192.168.137.244"
    borders = []
    classes = []
    filter = False
    def __init__(self, parent=None):
        QtCore.QThread.__init__(self, parent)

    def run(self):
        CHANNELS_TO_MONITOR = (1,)

        AD1299_NUM_CH = 8
        TRANSPORT_BLOCK_HEADER_SIZE = 16
        PKT_COUNT_OFFSET = 2
        SAMPLES_PER_TRANSPORT_BLOCK = 64
        TRANSPORT_QUE_SIZE = 4
        TCP_SERVER_PORT = 3000
        SPS = 1000
        SAMPLES_TO_COLLECT = SAMPLES_PER_TRANSPORT_BLOCK * 8 * 45

        TCP_PACKET_SIZE = int(
            ((TRANSPORT_BLOCK_HEADER_SIZE) / 4 + (AD1299_NUM_CH + 1) * (SAMPLES_PER_TRANSPORT_BLOCK)) * 4)

        # Create a TCP/IP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect the socket to the port where the server is listening
        server_address = (self.EMG8x_ADDRESS, TCP_SERVER_PORT)

        sock.connect(server_address)

        receivedBuffer = bytes()
        rec_data = bytes()
        self.rawSamples = np.zeros((SAMPLES_TO_COLLECT, len(CHANNELS_TO_MONITOR)))
        # Collected samples
        numSamples = 0
        is_predicated = False
        prd_array = np.array([])
        brd_array = np.array([])
        num_of_predications = 0
        self.borders = []
        self.classes = []
        lb = pickle.loads(open("lb.pickle", "rb").read())
        try:
            while True:
                if numSamples == (SAMPLES_TO_COLLECT - SAMPLES_PER_TRANSPORT_BLOCK) and not is_predicated:
                    self.signal.emit("Фильтрация сигнала и ожидание получения анализа...")
                    is_predicated = True
                    num_of_predications = sock.recv(3)
                    print("There are " + str(num_of_predications)[2] + str(num_of_predications)[3] + " predications")
                    num_of_predications = int(str(num_of_predications)[2] + str(num_of_predications)[3])
                    if num_of_predications == 0:
                        break
                    count = 0
                    while True:
                        prd_str = sock.recv(12)
                        prd = sock.recv(4 * 3)
                        brd = sock.recv(4 * 2)
                        rec_data += prd + brd
                        prd = np.frombuffer(prd, dtype=np.float32)
                        brd = np.frombuffer(brd, dtype=np.int32)
                        predication = np.array([prd])
                        cl = lb.classes_[predication.argmax(axis=1)[0]]
                        print(lb.classes_[predication.argmax(axis=1)[0]])
                        self.classes.append(cl)
                        self.borders.append(brd)
                        print(prd)
                        print(brd)
                        count = count + 1
                        if count == num_of_predications:
                            break
                if len(receivedBuffer) >= TCP_PACKET_SIZE * 2:
                    # find sync bytes
                    startOfBlock = receivedBuffer.find('EMG8x'.encode())
                    if startOfBlock >= 0:
                        strFormat = '{:d}i'.format(
                            round(SAMPLES_PER_TRANSPORT_BLOCK * (AD1299_NUM_CH + 1) + TRANSPORT_BLOCK_HEADER_SIZE / 4))
                        samples = unpack(strFormat, receivedBuffer[startOfBlock:startOfBlock + TCP_PACKET_SIZE])
                        # remove block from received buffer
                        receivedBuffer = receivedBuffer[startOfBlock + TCP_PACKET_SIZE:]
                        chCount = 0
                        for chIdx in CHANNELS_TO_MONITOR:
                            # get channel offset
                            offset_toch = int(TRANSPORT_BLOCK_HEADER_SIZE / 4 + SAMPLES_PER_TRANSPORT_BLOCK * chIdx)
                            dataSamples = samples[offset_toch:offset_toch + SAMPLES_PER_TRANSPORT_BLOCK]
                            blockSamples = np.array(dataSamples)
                            self.signal.emit('Ch#{0} Block #{1} mean:{2:10.1f},  var:{3:8.1f}, sec:{4:4.0f}'.format(chIdx, samples[
                                PKT_COUNT_OFFSET], np.mean(blockSamples), np.var(blockSamples) / 1e6, numSamples / SPS))
                            print('Ch#{0} Block #{1} mean:{2:10.1f},  var:{3:8.1f}, sec:{4:4.0f}'.format(chIdx, samples[
                                PKT_COUNT_OFFSET], np.mean(blockSamples), np.var(blockSamples) / 1e6, numSamples / SPS))
                            self.rawSamples[numSamples:numSamples + SAMPLES_PER_TRANSPORT_BLOCK, chCount] = blockSamples
                            chCount += 1
                        numSamples += SAMPLES_PER_TRANSPORT_BLOCK
                        if numSamples >= SAMPLES_TO_COLLECT:
                            break
                else:
                    receivedData = sock.recv(TCP_PACKET_SIZE)
                    if not receivedData:
                        break
                    receivedBuffer += receivedData
        finally:
            sock.close()
        self.sleep(1)


class WindowApp(QtWidgets.QMainWindow, window.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("EMG Predication App")
        navigation_toolbar = NavigationToolbar(self.mplWidget.canvas, self)
        #navigation_toolbar.
        self.addToolBar(navigation_toolbar)

        self.mplWidget.canvas.axes.grid(True)
        self.mplWidget.canvas.axes.set_xlabel("время, мс", size=12)
        self.mplWidget.canvas.axes.set_ylabel("напряжение", size=12 )
        self.timer = QtCore.QTimer()
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.update_plot)

        self.GetThread = GetThread()
        self.getSignal.clicked.connect(self.on_clicked)
        # self.getSignal.clicked.connect(self.fake_signal)
        self.ipButton.clicked.connect(self.change_ip)
        self.GetThread.started.connect(self.on_started)
        self.GetThread.finished.connect(self.on_finished)
        self.GetThread.signal.connect(self.on_change, QtCore.Qt.QueuedConnection)

    def fake_signal(self):
        x = np.loadtxt("1.txt")
        h = np.amax(np.abs(x))
        self.mplWidget.canvas.axes.clear()
        self.mplWidget.canvas.axes.grid(True)
        self.mplWidget.canvas.axes.plot(x)
        self.mplWidget.canvas.axes.vlines(3900, -h, h, color='yellow')
        self.mplWidget.canvas.axes.vlines(4500, -h, h, color='yellow')
        self.mplWidget.canvas.axes.hlines(-h, 3900, 4500, color='yellow')
        self.mplWidget.canvas.axes.vlines(9900, -h, h, color='green')
        self.mplWidget.canvas.axes.vlines(10500, -h, h, color='green')
        self.mplWidget.canvas.axes.hlines(-h, 9900, 10500, color='green')
        self.mplWidget.canvas.axes.vlines(18300, -h, h, color='red')
        self.mplWidget.canvas.axes.vlines(18900, -h, h, color='red')
        self.mplWidget.canvas.axes.hlines(-h, 18300, 18900, color='red')
        self.mplWidget.canvas.draw()

    def change_ip(self):
        self.GetThread.EMG8x_ADDRESS = self.ipLine.text()

    def on_clicked(self):
        self.timer.start()
        self.getSignal.setDisabled(True)  # Делаем кнопку неактивной
        self.GetThread.start()  # Запускаем поток

    def on_started(self):  # Вызывается при запуске потока
        self.label.setText("Программа запущена")

    def on_finished(self):  # Вызывается при завершении потока
        self.timer.stop()
        self.label.setText("Сигнал и ответ от устройства получены")
        self.getSignal.setDisabled(False)  # Делаем кнопку активной
        x = self.GetThread.rawSamples
        x -= np.mean(x)
        SPS = 1000.0
        x = np.array(x)
        x = np.reshape(x, len(x))
        hflt = signal.firls(513, [0., 5., 7., SPS / 2], [0., 0., 1.0, 1.0], fs=SPS)
        y = np.convolve(hflt, x, 'same')
        # y = x
        self.mplWidget.canvas.axes.clear()
        self.mplWidget.canvas.axes.grid(True)
        self.mplWidget.canvas.axes.plot(y)
        h = np.amax(np.abs(y))
        for i in range(len(self.GetThread.classes)):
            self.mplWidget.canvas.axes.text(self.GetThread.borders[i][0] + (self.GetThread.borders[i][1] -
                                                                            self.GetThread.borders[i][0]) // 2,
                                            (- h - 1000), self.GetThread.classes[i])
            if int(self.GetThread.classes[i]) == 1:
                self.mplWidget.canvas.axes.vlines(self.GetThread.borders[i][0], -h, h, color='red')
                self.mplWidget.canvas.axes.vlines(self.GetThread.borders[i][1], -h, h, color='red')
                self.mplWidget.canvas.axes.hlines(-h, self.GetThread.borders[i][0], self.GetThread.borders[i][1], color='red')
            if int(self.GetThread.classes[i]) == 2:
                self.mplWidget.canvas.axes.vlines(self.GetThread.borders[i][0], -h, h, color='yellow')
                self.mplWidget.canvas.axes.vlines(self.GetThread.borders[i][1], -h, h, color='yellow')
                self.mplWidget.canvas.axes.hlines(-h, self.GetThread.borders[i][0], self.GetThread.borders[i][1], color='yellow')
            if int(self.GetThread.classes[i]) == 3:
                self.mplWidget.canvas.axes.vlines(self.GetThread.borders[i][0], -h, h, color='green')
                self.mplWidget.canvas.axes.vlines(self.GetThread.borders[i][1], -h, h, color='green')
                self.mplWidget.canvas.axes.hlines(-h, self.GetThread.borders[i][0], self.GetThread.borders[i][1], color='green')
        self.mplWidget.canvas.axes.set_xlabel("время, мс", size=8)
        self.mplWidget.canvas.axes.set_ylabel("напряжение", size=8)
        self.mplWidget.canvas.draw()


    def on_change(self, s):
        self.label.setText(s)

    def update_plot(self):

        self.mplWidget.canvas.axes.clear()
        self.mplWidget.canvas.axes.grid(True)
        self.mplWidget.canvas.axes.plot(self.GetThread.rawSamples)
        self.mplWidget.canvas.axes.set_xlabel("время, мс", size=8)
        self.mplWidget.canvas.axes.set_ylabel("напряжение", size=8)
        self.mplWidget.canvas.draw()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = WindowApp()
    apply_stylesheet(app, theme='light_blue.xml')
    window.show()
    app.exec_()


if __name__ == "__main__":
    # print(socket.gethostname() )
    # print(socket.gethostbyname(socket.gethostname() ))
    main()
# pyuic5
