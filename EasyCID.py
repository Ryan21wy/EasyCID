import os
import sys
import tensorflow as tf

if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

import webbrowser
import sqlite3
import numpy as np
import pandas as pd
import json
import matplotlib
import chardet

matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')

from tensorflow.python.framework import ops
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QAbstractItemView, \
    QFileDialog, QDialog, QMessageBox, QTreeWidgetItem, QAction, QMenu, QHeaderView
from PyQt5.QtCore import QThread, QStringListModel, pyqtSignal
from PyQt5.QtGui import QIcon, QStandardItemModel, QStandardItem
from matplotlib.figure import Figure
from sklearn.linear_model import enet_path
# local package
from spc.spcio import readSPC
import jcamp
from AirPLS import airPLS, WhittakerSmooth
# windows of EasyCID
from EasyCID_MainWindow import Ui_MainWindow
from child_win import TrainNewP_win
from child_win import TableName_win
from child_win import ModelSelect_win
from child_win import RatioEstimation_win
from child_win import TrainHistory_win
from child_win import LinkModels_win


def locate_line(lines, keyword):
    for i in range(0, len(lines)):
        fr = lines[i].find(keyword)
        if fr != -1:
            return i
    return "CAN'T FIND";


def parseBWTekFile(filepath, filename, select_ramanshift=False, xname='Raman Shift', yname="Dark Subtracted #1"):
    file_content = open(filepath).read()
    fc_splits = file_content.split('\n')
    spectrum = {}
    spectrum['name'] = filename[0:(len(filename) - 4)]
    spectrum['excition'] = float(fc_splits[locate_line(fc_splits, 'laser_wavelength')].split(';')[1])
    spectrum['integral_time'] = float(fc_splits[locate_line(fc_splits, 'intigration times')].split(';')[1])

    nTitle = locate_line(fc_splits, xname)

    Xunit = fc_splits[nTitle].split(';')
    AXI = Xunit.index(xname)
    DSI = Xunit.index(yname)
    xaxis_min = nTitle + 1 + int(fc_splits[locate_line(fc_splits, 'xaxis_min')].split(';')[1])
    xaxis_max = nTitle + 1 + int(fc_splits[locate_line(fc_splits, 'xaxis_max')].split(';')[1])

    nLen = xaxis_max - xaxis_min
    spectrum["axis"] = np.zeros((nLen,), dtype=np.float64)
    spectrum["spectrum"] = np.zeros((nLen,), dtype=np.float64)
    for i in range(xaxis_min, xaxis_max):
        fc_ss = fc_splits[i].split(';')
        spectrum["axis"][i - xaxis_min] = float(fc_ss[AXI])
        spectrum["spectrum"][i - xaxis_min] = float(fc_ss[DSI])
    if select_ramanshift:
        inds = np.logical_and(spectrum["axis"] > 160, spectrum["axis"] < 3000)
        spectrum["axis"] = spectrum["axis"][inds]
        spectrum["spectrum"] = spectrum["spectrum"][inds].astype(np.float64)
    return spectrum


def cnn_model(x_train):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(input_shape=(x_train.shape[1], 1), filters=32, kernel_size=(3), strides=(2),
                               padding='SAME', kernel_initializer=tf.keras.initializers.he_normal(), use_bias=False),
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool1D(pool_size=(2)),
        tf.keras.layers.Conv1D(input_shape=(x_train.shape[1] / 2, 32), filters=64, kernel_size=(3), strides=(2),
                               padding='SAME', kernel_initializer=tf.keras.initializers.he_normal(), use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool1D(pool_size=(2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.he_normal()),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


class TrainNewP(QDialog):
    signal_parp = pyqtSignal(list)

    def __init__(self, GroupMI=None, ComponentMI=None, fixed=False):
        QDialog.__init__(self)
        self.child = TrainNewP_win.Ui_Dialog()
        self.child.setupUi(self)
        self.child.dir_choose.setIcon(QIcon('./EasyCID_Icon/view.png'))
        self.child.aug_choose.setIcon(QIcon('./EasyCID_Icon/view.png'))
        self.child.dir_choose.clicked.connect(self.dir_chose)
        self.child.aug_choose.clicked.connect(self.aug_chose)
        self.child.save_.clicked.connect(self.signal_emit)
        self.child.cancel_.clicked.connect(self.cancel)
        self.child.save_.setEnabled(True)
        self.signal = []
        if GroupMI:
            self.child.startshift.setValue(GroupMI[0])
            self.child.endshift.setValue(GroupMI[1])
            self.child.interval.setValue(GroupMI[2])
            self.child.aug_savepath.setText(GroupMI[3])
            self.child.savepath.setText(GroupMI[4])
        if ComponentMI:
            self.child.number.setValue(ComponentMI[0])
            nr1, nr2 = self.extract_num(ComponentMI[1])
            self.child.noise1.setValue(nr1)
            self.child.noise2.setValue(nr2)
            self.child.optimizer_.setCurrentIndex(ComponentMI[2])
            lr1, lr2 = self.extract_num(ComponentMI[3])
            self.child.lr1.setValue(lr1)
            self.child.lr2.setValue(lr2)
            self.child.batchsize.setValue(ComponentMI[4])
            self.child.epochs_.setValue(ComponentMI[5])
        if fixed:
            self.child.startshift.setEnabled(False)
            self.child.endshift.setEnabled(False)
            self.child.interval.setEnabled(False)
            self.child.dir_choose.setEnabled(False)

    def extract_num(self, num):
        int_ = abs(int(np.log10(num))) + 1
        b0 = 10 ** -int_
        min_ = num / b0
        if min_ >= 10:
            int_ = int_ - 1
            min_ = 1.00
        return min_, int_

    def dir_chose(self):
        last = self.child.savepath.text()
        if last:
            save_path = QFileDialog.getExistingDirectory(win.centralwidget, "choose save path", last)
        else:
            save_path = QFileDialog.getExistingDirectory(win.centralwidget, "choose save path", "C:/")
        self.child.savepath.setText(save_path)

    def aug_chose(self):
        last = self.child.aug_savepath.text()
        if last:
            save_path = QFileDialog.getExistingDirectory(win.centralwidget, "choose save path", last)
        else:
            save_path = QFileDialog.getExistingDirectory(win.centralwidget, "choose save path", "C:/")
        self.child.aug_savepath.setText(save_path)

    def signal_emit(self):
        self.signal = []
        start_shift = self.child.startshift.value()
        end_shift = self.child.endshift.value()
        interval = self.child.interval.value()
        if end_shift <= start_shift:
            QMessageBox.warning(self, "error", 'The end of Raman shift cannot be smaller than the start')
            return
        elif (end_shift - start_shift) < interval:
            QMessageBox.warning(self, "error", 'The interval is too big')
            return
        self.signal.append(start_shift)
        self.signal.append(end_shift)
        self.signal.append(interval)
        self.signal.append(self.child.optimizer_.currentIndex())
        lr1 = self.child.lr1.value()
        lr2 = self.child.lr2.value()
        lr = lr1 * 10 ** (-lr2)
        self.signal.append(lr)
        self.signal.append(self.child.batchsize.value())
        self.signal.append(self.child.epochs_.value())
        self.signal.append(self.child.number.value())
        nr1 = self.child.noise1.value()
        nr2 = self.child.noise2.value()
        nr = nr1 * 10 ** (-nr2)
        self.signal.append(nr)
        self.signal.append(self.child.aug_savepath.text())
        if not self.child.savepath.text():
            QMessageBox.warning(self, "error", 'The save path cannot be empty!')
            return
        self.signal.append(self.child.savepath.text())
        self.signal_parp.emit(self.signal)
        TrainNewP.close(self)

    def cancel(self):
        TrainNewP.close(self)


class TrainReport(QDialog):
    def __init__(self, parameters):
        QDialog.__init__(self)
        self.child = TrainHistory_win.Ui_Dialog()
        self.child.setupUi(self)
        self.parameters = parameters
        self.names = list(parameters.keys())
        self.child.tableView.clicked.connect(self.plot)
        self.child.pushButton.clicked.connect(self.close)
        self.fig = Myplot(dpi=100)
        self.fig_ntb = NavigationToolbar(self.fig, self)
        self.ax1 = self.fig.axes
        self.ax2 = self.ax1.twinx()
        self.gridlayout = QGridLayout(self.child.groupBox)
        self.gridlayout.addWidget(self.fig)
        self.gridlayout.addWidget(self.fig_ntb)
        TrainReport.load(self)

    def load(self):
        self.model = QStandardItemModel(len(self.names), 3)
        self.model.setHorizontalHeaderLabels(['Component', 'Test Loss', 'Test Acc'])
        for r in range(len(self.names)):
            it_1 = QStandardItem(self.names[r])
            self.model.setItem(r, 0, it_1)
            it_2 = QStandardItem('%.3g' % self.parameters[self.names[r]][4])
            self.model.setItem(r, 1, it_2)
            it_3 = QStandardItem('%.3g' % self.parameters[self.names[r]][5])
            self.model.setItem(r, 2, it_3)
        self.child.tableView.setModel(self.model)
        self.child.tableView.horizontalHeader().setStretchLastSection(True)
        self.child.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def plot(self):
        index = self.child.tableView.currentIndex().row()
        name = self.names[index]
        t_loss = self.parameters[name][0]
        t_acc = self.parameters[name][1]
        v_loss = self.parameters[name][2]
        v_acc = self.parameters[name][3]
        x_axe = np.arange(1, len(t_loss) + 1)
        self.ax1.cla()
        self.ax2.cla()
        self.ax1.set_xlabel("Epoch", fontsize=12, color='k')
        self.ax1.set_ylabel("Loss", fontsize=12, color='g')
        self.ax2.set_ylabel("Accuracy", fontsize=12, color='r')
        lns1 = self.ax1.plot(x_axe, t_loss, label='train loss', c='g', linewidth=1)
        lns2 = self.ax1.plot(x_axe, v_loss, label='validation loss', linestyle=':', c='g', linewidth=1.5)
        lns3 = self.ax2.plot(x_axe, t_acc, label='train acc', c='r', linewidth=1)
        lns4 = self.ax2.plot(x_axe, v_acc, label='validation acc', linestyle=':', c='r', linewidth=1.5)
        lns = lns1 + lns2 + lns3 + lns4
        labs = [l.get_label() for l in lns]
        self.ax1.legend(lns, labs, loc=7)
        self.ax1.set_title(name, fontsize=15, color='b')
        self.fig.draw()


class ChangeName(QDialog):
    signal_parp = pyqtSignal(str)

    def __init__(self, init_name):
        QDialog.__init__(self)
        self.child = TableName_win.Ui_Dialog()
        self.child.setupUi(self)
        self.init_name = init_name
        self.child.name.setText(init_name)
        self.child.save_.clicked.connect(self.signal_emit)
        self.child.reset_.clicked.connect(self.value_reset)

    def signal_emit(self):
        name = self.child.name.text()
        self.signal_parp.emit(name)
        ChangeName.close(self)

    def value_reset(self):
        self.child.name.setText(self.init_name)


class LinkModels(QDialog):
    signal_parp = pyqtSignal(list)

    def __init__(self, old_para=None):
        QDialog.__init__(self)
        self.child = LinkModels_win.Ui_Dialog()
        self.child.setupUi(self)
        if len(old_para) > 1:
            self.child.startshift.setValue(old_para[0])
            self.child.endshift.setValue(old_para[1])
            self.child.interval.setValue(old_para[2])
        self.child.dir_choose.setIcon(QIcon('./EasyCID_Icon/view.png'))
        self.child.dir_choose.clicked.connect(self.dir_chose)
        print(old_para)
        self.group = old_para[-1]
        self.child.link_.clicked.connect(self.link)
        self.child.cancel_.clicked.connect(self.cancel)

    def dir_chose(self):
        last = self.child.modelpath.text()
        if last:
            model_path = QFileDialog.getExistingDirectory(win.centralwidget, "choose model path", last)
        else:
            model_path = QFileDialog.getExistingDirectory(win.centralwidget, "choose model path", "C:/")
        self.child.modelpath.setText(model_path)

    def link(self):
        signal = []
        model_path = self.child.modelpath.text()
        if not model_path:
            return
        start_shift = self.child.startshift.value()
        end_shift = self.child.endshift.value()
        interval = self.child.interval.value()
        if end_shift <= start_shift:
            QMessageBox.warning(self, "error", 'The end of Raman shift cannot be smaller than the start')
            return
        elif (end_shift - start_shift) < interval:
            QMessageBox.warning(self, "error", 'The interval is too big')
            return
        correct_models = []
        x_test = np.arange(start_shift, end_shift, interval).reshape(1, -1)
        reload_model = cnn_model(x_test)
        tf.keras.backend.clear_session()
        ops.reset_default_graph()
        dir = os.listdir(model_path)
        for file in dir:
            path = os.path.join(model_path, file)
            if os.path.isfile(path) and file.split('.')[-1] == 'h5':
                try:
                    reload_model.load_weights(path)
                    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
                    reload_model.predict(x_test)
                except Exception as err:
                    QMessageBox.warning(self, "error", str(err))
                    return
                correct_models.append(file.split('.')[0])
        if not correct_models:
            QMessageBox.warning(self, "error", 'No available models were found')
            return
        signal.append(self.group)
        signal.append(start_shift)
        signal.append(end_shift)
        signal.append(interval)
        signal.append(model_path)
        signal.append(correct_models)
        self.signal_parp.emit(signal)
        LinkModels.close(self)

    def cancel(self):
        LinkModels.close(self)


class RatioEstimation(QDialog):
    signal_parp = pyqtSignal(dict)

    def __init__(self):
        QDialog.__init__(self)
        self.child = RatioEstimation_win.Ui_dialog()
        self.child.setupUi(self)

        self.widget_1(False)
        self.widget_2(False)

        self.child.OK.clicked.connect(self.signal_emit)
        self.child.checkBox.stateChanged.connect(self.checkBox_state)
        self.child.checkBox_2.stateChanged.connect(self.checkBox_2_state)
        self.child.cancel_.clicked.connect(self.cancel)

    def widget_1(self, state=False):
        self.child.label_2.setEnabled(state)
        self.child.label_3.setEnabled(state)
        self.child.label_4.setEnabled(state)
        self.child.sb_lamda.setEnabled(state)
        self.child.sb_maxiter.setEnabled(state)
        self.child.sb_proder.setEnabled(state)

    def widget_2(self, state=False):
        self.child.label_5.setEnabled(state)
        self.child.sm_lamda.setEnabled(state)

    def checkBox_state(self):
        if self.child.checkBox.isChecked():
            self.widget_1(True)
        else:
            self.widget_1(False)

    def checkBox_2_state(self):
        if self.child.checkBox_2.isChecked():
            self.widget_2(True)
        else:
            self.widget_2(False)

    def signal_emit(self):
        param = {'sb': None, 'sm': None, 'en': None}
        if self.child.checkBox.isChecked():
            sb_param = []
            sb_param.append(self.child.sb_lamda.text())
            sb_param.append(self.child.sb_proder.text())
            sb_param.append(self.child.sb_maxiter.text())
            param['sb'] = sb_param
        if self.child.checkBox_2.isChecked():
            param['sm'] = self.child.sm_lamda.text()
        en_param = []
        en_param.append(self.child.EN_ratio.text())
        en_param.append(self.child.EN_iter.text())
        param['en'] = en_param
        self.signal_parp.emit(param)
        RatioEstimation.close(self)

    def cancel(self):
        RatioEstimation.close(self)


class PredictionSetting(QDialog):
    signal_parp = pyqtSignal(list)

    def __init__(self, table_list):
        QDialog.__init__(self)
        self.child = ModelSelect_win.Ui_dialog()
        self.child.setupUi(self)
        for item in table_list:
            self.child.comboBox.addItem(item)
        self.child.OK.clicked.connect(self.signal_emit)
        self.child.cancel_.clicked.connect(self.cancel)

    def signal_emit(self):
        signal = []
        table = self.child.comboBox.currentText()
        threshold = self.child.Threshold.value()
        signal.append(table)
        signal.append(threshold)
        self.signal_parp.emit(signal)
        PredictionSetting.close(self)

    def cancel(self):
        PredictionSetting.close(self)


class TrainRun(QThread):
    process_signal = pyqtSignal(str)
    signal = pyqtSignal(str)
    err_signal = pyqtSignal(str)
    data_signal = pyqtSignal(list)
    para_signal = pyqtSignal(list)

    def __init__(self, train_para, aug_para, info_para):
        super().__init__()
        self.optimizer = train_para[0]
        self.batch_size = train_para[1]
        self.epochs = train_para[2]
        self.model_path = train_para[3]
        self.aug_number = aug_para[0]
        self.noise_rate = aug_para[1]
        self.aug_save_path = aug_para[2]
        self.names = info_para[0]
        self.count = info_para[1]
        self.spectra = info_para[2]
        self.axis = info_para[3]
        self.new_axis = info_para[4]

    def load_data(self, X, Y1):
        for i in range(X.shape[0]):
            X[i, :] = (X[i, :] - np.min(X[i, :])) / (np.max(X[i, :]) - np.min(X[i, :]))
        Xtrain = X[0:int(0.9 * X.shape[0])]
        Xtest = X[int(0.9 * X.shape[0]):X.shape[0]]
        Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1], 1)
        Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1], 1)

        Ytrain = Y1[0:int(0.9 * X.shape[0])]
        Ytest = Y1[int(0.9 * X.shape[0]):X.shape[0]]
        return Xtrain, Xtest, Ytrain, Ytest

    def noise(self, spectrum, nr):
        N_MAX = nr * np.max(spectrum)
        Xnoise = np.random.normal(0, N_MAX, (1, spectrum.shape[1]))
        spectrum_noise = spectrum + Xnoise
        return np.maximum(spectrum_noise, 0)

    def component_in(self, spectrum_raw, component, num, nr=0):
        c1 = np.random.uniform(0.1, 1, (num, 1))
        c2 = np.random.rand(num, 1)
        k = np.random.randint(0, spectrum_raw.shape[0], size=(num, 1))
        Spectrumdata_new = np.zeros((1, spectrum_raw.shape[1]))
        component_in = (spectrum_raw[component, :]).reshape((1, spectrum_raw.shape[1]))
        for i in range(num):
            Spectrumdata_new2 = c1[i] * component_in + c2[i] * spectrum_raw[k[i], :]
            Spectrumdata_new2 = TrainRun.noise(self, Spectrumdata_new2, nr)
            Spectrumdata_new = np.vstack((Spectrumdata_new, Spectrumdata_new2))
        Spectrumdata_new = np.delete(Spectrumdata_new, 0, 0)
        label = np.ones((num, 1))
        return {'spectrum_in': Spectrumdata_new, 'label_in': label}

    def component_out(self, spectrum_raw, component, num, nr=0):
        c1 = np.random.rand(num, 1)
        c2 = np.random.rand(num, 1)
        k = np.random.randint(0, spectrum_raw.shape[0], size=(num, 1))
        for j in range(num):
            while k[j] == component:
                k[j] = np.random.randint(0, spectrum_raw.shape[0])
        h = np.random.randint(0, spectrum_raw.shape[0], size=(num, 1))
        for l in range(num):
            while h[l] == component:
                h[l] = np.random.randint(0, spectrum_raw.shape[0])
        Spectrumdata_new = np.zeros((1, spectrum_raw.shape[1]))
        for i in range(num):
            Spectrumdata_new2 = c1[i] * spectrum_raw[k[i]] + c2[i] * spectrum_raw[h[i]]
            Spectrumdata_new2 = TrainRun.noise(self, Spectrumdata_new2, nr)
            Spectrumdata_new = np.vstack((Spectrumdata_new, Spectrumdata_new2))
        Spectrumdata_new = np.delete(Spectrumdata_new, 0, 0)
        label = np.zeros((num, 1))
        return {'spectrum_out': Spectrumdata_new, 'label_out': label}

    def randomize(self, dataset, labels):
        permutation = np.random.permutation(labels.shape[0])
        shuffled_dataset = dataset[permutation, :]
        shuffled_labels = labels[permutation]
        return shuffled_dataset, shuffled_labels

    def run(self):
        try:
            self.signal.emit('run')
            Spectrumdata = np.zeros((1, self.new_axis.shape[0]))
            count = len(self.spectra)
            new_spectra = []
            for i in range(count):
                spectrum = np.interp(self.new_axis, self.axis[i], self.spectra[i]).astype(np.float64).copy()
                new_spectra.append([self.names[i], spectrum])
                # spectrum = spectrum / np.max(spectrum)
                Spectrumdata = np.vstack((Spectrumdata, spectrum.T))
            spectra_raw = np.delete(Spectrumdata, 0, 0)
            a = 1
            for num in self.count:
                para_record = []
                self.process_signal.emit('Augmentation Active (%s/%s) :' % (str(a), str(len(self.count))))
                if self.aug_save_path:
                    mkdir(self.aug_save_path)
                    aug_data_path = os.path.join(self.aug_save_path, self.names[num] + '.npy')
                    aug_label_path = os.path.join(self.aug_save_path, self.names[num] + '_label.npy')
                    if os.path.isfile(aug_data_path):
                        spectrum = np.load(aug_data_path)
                        label = np.load(aug_label_path)
                    else:
                        num_sample = self.aug_number
                        data_in = TrainRun.component_in(self, spectra_raw, num,
                                                        num=int(num_sample / 2), nr=self.noise_rate)
                        Xin = data_in['spectrum_in']
                        Yin = data_in['label_in']
                        data_out = TrainRun.component_out(self, spectra_raw, num,
                                                          num=int(num_sample / 2), nr=self.noise_rate)
                        Xout = data_out['spectrum_out']
                        Yout = data_out['label_out']
                        spectrum = np.concatenate((Xin, Xout), axis=0)
                        label = np.concatenate((Yin, Yout), axis=0)
                        np.save(aug_data_path, spectrum)
                        np.save(aug_label_path, label)
                else:
                    num_sample = self.aug_number
                    data_in = TrainRun.component_in(self, spectra_raw, num,
                                                    num=int(num_sample / 2), nr=self.noise_rate)
                    Xin = data_in['spectrum_in']
                    Yin = data_in['label_in']
                    data_out = TrainRun.component_out(self, spectra_raw, num,
                                                      num=int(num_sample / 2), nr=self.noise_rate)
                    Xout = data_out['spectrum_out']
                    Yout = data_out['label_out']
                    spectrum = np.concatenate((Xin, Xout), axis=0)
                    label = np.concatenate((Yin, Yout), axis=0)
                spectrumdata, labeldata = TrainRun.randomize(self, spectrum, label)
                Xtrain, Xtest, Ytrain, Ytest = TrainRun.load_data(self, spectrumdata, labeldata)
                self.process_signal.emit('Training Active (%s/%s) :' % (str(a), str(len(self.count))))
                tf.keras.backend.clear_session()
                ops.reset_default_graph()

                model = cnn_model(Xtrain)
                model.compile(optimizer=self.optimizer,
                              loss='binary_crossentropy',
                              metrics=['accuracy'])
                model.summary()

                callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                             min_delta=0.0001, patience=20, verbose=0, mode='auto',
                                                             baseline=None, restore_best_weights=True)]

                history = model.fit(Xtrain, Ytrain, batch_size=self.batch_size, epochs=self.epochs,
                                    validation_split=0.1, verbose=0, callbacks=callback)
                loss, acc = model.evaluate(Xtest, Ytest, verbose=0)
                para_record.append(self.names[num])
                para_record.append(history.history['loss'])
                para_record.append(history.history['accuracy'])
                para_record.append(history.history['val_loss'])
                para_record.append(history.history['val_accuracy'])
                para_record.append(loss)
                para_record.append(acc)
                mkdir(self.model_path)
                model.save_weights(os.path.join(self.model_path, self.names[num] + '.h5'))

                del model
                self.data_signal.emit(new_spectra[num])
                self.para_signal.emit(para_record)
                a += 1
            self.signal.emit('finished')
        except Exception as err:
            self.err_signal.emit(str(err))


class PredRun(QThread):
    signal = pyqtSignal(str)
    data_signal = pyqtSignal(list)

    def __init__(self, x_test, model_path):
        super().__init__()
        self.model_path = model_path
        self.x_test = x_test

    def run(self):
        try:
            self.signal.emit('run')
            Xtest_pre = np.zeros(self.x_test.shape)
            for i in range(self.x_test.shape[0]):
                Xtest_pre[i, :] = self.x_test[i, :] / np.max(self.x_test[i, :])
            reload_model = cnn_model(Xtest_pre)
            reload_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                                 loss='binary_crossentropy',
                                 metrics=['accuracy'])
            y_DeepCID = []
            for path in self.model_path:
                tf.keras.backend.clear_session()
                ops.reset_default_graph()
                reload_model.load_weights(path)
                Xtest_pre = Xtest_pre.reshape(Xtest_pre.shape[0], Xtest_pre.shape[1], 1)
                y = reload_model.predict(Xtest_pre)
                y_DeepCID.append(y)
            self.data_signal.emit(y_DeepCID)
            self.signal.emit('finished')
        except Exception as err:
            self.signal.emit(str(err))


class QARun(QThread):
    signal = pyqtSignal(str)
    rate_signal = pyqtSignal(list)

    def __init__(self, mix, com, parameters):
        super().__init__()
        self.mix = mix
        self.com = com
        self.parameters = parameters

    def run(self):
        try:
            self.signal.emit('run')
            sb_param = self.parameters['sb']
            sm_param = self.parameters['sm']
            en_param = self.parameters['en']
            ratios = []
            k = 0
            for m in self.mix:
                com_spectra = self.com[k]
                if sm_param is not None:
                    for i in range(com_spectra.shape[0]):
                        com_spectra[i] = WhittakerSmooth(com_spectra[i], np.ones(com_spectra[i].shape[0]),
                                                         int(sm_param))
                if sb_param is not None:
                    for i in range(com_spectra.shape[0]):
                        com_spectra[i] = com_spectra[i] - airPLS(com_spectra[i], int(sb_param[0]), int(sb_param[1]),
                                                                 int(sb_param[2]))
                _, coefs_lasso, _ = enet_path(com_spectra.T, m, l1_ratio=float(en_param[0]),
                                              n_alphas=int(en_param[1]), positive=True, fit_intercept=False)
                ratio = coefs_lasso[:, -1]
                ratio = ratio / sum(ratio)
                ratio[-1] = 1 - sum(ratio[:-1])
                ratio = np.round(ratio, 3)
                ratios.append(ratio)
                k += 1
            self.rate_signal.emit(ratios)
            self.signal.emit('finished')
        except Exception as err:
            self.signal.emit(str(err))


class EXCELCreate(QThread):
    signal = pyqtSignal(str)

    def __init__(self, pred_names, pred_list, c_list, save_path):
        super().__init__()
        self.pred_names = pred_names
        self.pred_list = pred_list
        self.save_path = save_path
        self.c_list = c_list

    def run(self):
        try:
            self.signal.emit('run')
            data = []
            if self.c_list:
                for i in range(len(self.pred_names)):
                    mix = self.pred_names[i]
                    for j in range(len(self.pred_list[mix])):
                        if j == 0:
                            data.append([mix, self.pred_list[mix][j], self.c_list[i][j]])
                        else:
                            data.append([' ', self.pred_list[mix][j], self.c_list[i][j]])
                df = pd.DataFrame(data, columns=['Mixture', 'Component', 'Ratio'])
            else:
                for i in range(len(self.pred_names)):
                    mix = self.pred_names[i]
                    for j in range(len(self.pred_list[mix])):
                        if j == 0:
                            data.append([mix, self.pred_list[mix][j]])
                        else:
                            data.append([' ', self.pred_list[mix][j]])
                df = pd.DataFrame(data, columns=['mixture', 'component'])
            df.to_excel(self.save_path, index=False)
            self.signal.emit('finish')
        except Exception as err:
            self.signal.emit(str(err))


class Myplot(FigureCanvas):
    def __init__(self, dpi=100):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        self.fig = Figure(dpi=dpi, tight_layout=True)
        FigureCanvas.__init__(self, self.fig)
        self.axes = self.fig.add_subplot(111)
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class AppWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(AppWindow, self).__init__(parent)
        self.setupUi(self)
        action_ = self.addToolBar('Action')
        action_.setStyleSheet("background-color: rgb(240, 240, 240);")
        clear_ = QAction(QIcon("./EasyCID_Icon/clear.png"), 'Clear plot area', self)
        clear_.triggered.connect(self.erase_plot_func)
        action_.addAction(clear_)
        collect_ = QAction(QIcon('./EasyCID_Icon/collect.png'), 'Multiple plot', self)
        collect_.triggered.connect(self.mutiplot_func)
        action_.addAction(collect_)
        func_ = self.addToolBar('Function')
        add_ = QAction(QIcon('./EasyCID_Icon/add.png'), 'Train CNN models', self)
        add_.triggered.connect(self.train_models)
        func_.addAction(add_)
        run_ = QAction(QIcon('./EasyCID_Icon/predict.png'), 'Prediction', self)
        run_.triggered.connect(self.predict_process_func)
        func_.addAction(run_)
        self.qa = QAction(QIcon('./EasyCID_Icon/analysis.png'), 'Quantitative analysis', self)
        self.qa.triggered.connect(self.QA_process_func)
        func_.addAction(self.qa)

        self.fig = Myplot(dpi=100)
        self.fig.axes.set_xlabel("Raman Shift", fontsize=12, color='k')
        self.fig.axes.set_ylabel("Intensity", fontsize=12, color='k')
        self.fig_ntb = NavigationToolbar(self.fig, self)
        self.gridlayout_1 = QGridLayout(self.data_plot)
        self.gridlayout_1.addWidget(self.fig)
        self.gridlayout_1.addWidget(self.fig_ntb)
        self.fig_2 = Myplot(dpi=100)
        self.fig_2.axes.set_xlabel("Raman Shift", fontsize=12, color='k')
        self.fig_2.axes.set_ylabel("Intensity", fontsize=12, color='k')
        self.fig_ntb_2 = NavigationToolbar(self.fig_2, self.component_plot)
        self.gridlayout_1 = QGridLayout(self.component_plot)
        self.gridlayout_1.addWidget(self.fig_2)
        self.gridlayout_1.addWidget(self.fig_ntb_2)

        self.data_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.data_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.data_list.doubleClicked.connect(self.click_to_plot_mix)

        self.open_act.triggered.connect(self.open_dir_func)
        self.import_act.triggered.connect(self.load_spectra)
        self.save_act.triggered.connect(self.save_function)
        self.action_link.triggered.connect(self.link_to_db)
        self.action_setup.triggered.connect(self.set_up_db)
        self.menuHelp.triggered.connect(self.help_html)

        self.data_display.doubleClicked.connect(self.click_to_plot)
        self.data_display.setContextMenuPolicy(Qt.CustomContextMenu)
        self.data_display.setColumnCount(2)
        self.data_display.setHeaderLabels(['Component', ' Trained '])
        self.data_display.header().setStretchLastSection(False)
        self.data_display.header().setSectionResizeMode(QHeaderView.Stretch)
        self.data_display.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.data_display.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.data_display.customContextMenuRequested.connect(self.data_display_menu)

        self.predict_result.itemChanged.connect(self.get_checked)
        self.predict_result.setColumnCount(2)
        self.predict_result.setHeaderLabels([' Component ', ' Ratio '])
        self.predict_result.header().setStretchLastSection(False)
        self.predict_result.header().setSectionResizeMode(QHeaderView.Stretch)
        self.predict_result.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)

        self.modelPath.setTitle('')
        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(False)
        self.progressBar_2.setValue(0)
        self.progressBar_2.setTextVisible(False)
        self.textBrowser_3.setText('')
        self.textBrowser_4.setText('')

        self.mix_data = {}
        self.train_index = []
        self.mix_list = []
        self.result_list = []
        self.axis = None
        self.t_axis = []
        self.t_new_axis = None
        self.train_group = ''
        self.pred_group = ''
        self.threshold = 0.5
        self.pred_names = []
        self.pred_data = []
        self.pred_param = {}
        self.data_name = []
        self.c_data_list = []
        self.raman_shift_para = []
        self.train_com_name = []
        self.train_com_spec = []
        self.train_para = {}
        self.ratios = []
        self.muti = False
        self.pred_on = False
        self.train_on = False
        self.QA_on = False
        self.model_path = ''
        self.data_path = ''
        self.add_spectra_path = ''
        self.plot_lock = False
        self.db = None
        self.cur = None
        self.GroupMI = None
        self.ComponentMI = None
        self.link_widget = None
        self.qa.setEnabled(False)
        self.model_ref = {1: 'Yes', 0: 'No'}
        if os.path.exists('./EasyCID.db'):
            AppWindow.connect_db(self, './EasyCID.db')

        if tf.config.experimental.list_physical_devices('GPU'):
            gpus = tf.config.list_physical_devices(device_type='GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(device=gpu, enable=True)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    def help_html(self):
        path = 'file:///' + os.path.abspath('./documentation/html/index.html')
        webbrowser.open_new_tab(path)

    def connect_db(self, path):
        try:
            self.db = sqlite3.connect(path)
            self.db.execute('pragma foreign_keys=on')
            self.cur = self.db.cursor()
            abs_bath = os.path.abspath(path)
            self.modelPath.setTitle('Current Database: "%s"' % abs_bath)
            AppWindow.db_data_display(self)
        except Exception as err:
            QMessageBox.information(self, "Error", str(err))

    def set_up_db(self):
        file_name, _ = QFileDialog.getSaveFileName(self.centralwidget, "Build database", "C:/", 'database (*.db)')
        if not file_name:
            return
        else:
            db = sqlite3.connect(file_name)
            cur = db.cursor()
            sql = 'CREATE TABLE Groups (Group_ID INTEGER PRIMARY KEY, Group_Name VARCHAR)'
            cur.execute(sql)
            sql = 'CREATE TABLE Component_Info (Component_ID INTEGER PRIMARY KEY, Component_Name VARCHAR, ' \
                  'Raw_Spectrum BLOB, Raw_Axis BLOB, Inter_Time FLOAT, Model INTEGER, From_Group INTEGER, ' \
                  'foreign key(From_Group) references Groups(Group_ID) on delete cascade on update cascade) '
            cur.execute(sql)
            sql = 'CREATE TABLE Group_Model_Info (Raman_Start FLOAT, Raman_End FLOAT, Raman_Interval FLOAT, ' \
                  'Aug_Save_Path VARCHAR, Save_Path VARCHAR, From_Group INTEGER, foreign key(From_Group) references ' \
                  'Groups(Group_ID) on delete cascade on update cascade) '
            cur.execute(sql)
            sql = 'CREATE TABLE Component_Model_Info (Augment_Num INTEGER, Noise_Rate FLOAT, Optimizer INTEGER, ' \
                  'LR FLOAT, BS INTEGER, EPS INTEGER, From_Component INTEGER, foreign key(From_Component) references ' \
                  'Component_Info(Component_ID) on delete cascade on update cascade) '
            cur.execute(sql)
            db.commit()
        AppWindow.connect_db(self, file_name)

    def link_to_db(self):
        file_name, type = QFileDialog.getOpenFileName(self.centralwidget, "Choose database",
                                                      r"C:/", 'database (*.db)')
        if not file_name:
            return
        AppWindow.connect_db(self, file_name)

    def get_db_data(self, cur):
        sql = "select * from Groups"
        cur.execute(sql)
        group_db = cur.fetchall()
        group_names = [line[1] for line in group_db]
        group_ids = [line[0] for line in group_db]
        if not group_names:
            return None, None
        components_info = []
        for id in group_ids:
            sql = 'select Component_Name,Model from Component_Info where From_Group=?'
            component_info = cur.execute(sql, (id,)).fetchall()
            components_info.append(component_info)
        return group_names, components_info

    def db_data_display(self):
        if not self.db:
            return
        group_names, components_info = AppWindow.get_db_data(self, self.cur)
        if group_names is None:
            self.data_display.clear()
            return
        self.data_display.clear()
        for i in range(len(group_names)):
            group_name = group_names[i]
            component_info = components_info[i]
            root = QTreeWidgetItem(self.data_display)
            root.setText(0, group_name)
            for column in component_info:
                child = QTreeWidgetItem()
                child.setText(0, column[0])
                child.setText(1, self.model_ref[column[1]])
                root.addChild(child)
        self.data_display.expandAll()
        self.tabWidget.setCurrentIndex(0)

    def read_simple_txt(self, file_path):
        x = []
        y = []
        try:
            f = open(file_path, 'rb')
            r = f.read()
            charInfo = chardet.detect(r)
            f = open(file_path, 'r', encoding=(charInfo['encoding']))
            d = f.readlines()

            for line in d:
                d1 = line.replace(',', ' ').split()
                try:
                    x.append(float(d1[0]))
                    y.append(float(d1[1]))
                except:
                    pass
            return np.array(x), np.array(y)
        except Exception as err:
            QMessageBox.information(self, "Error", str(err))
            return None, None

    def open_dir_func(self):
        self.data_path = QFileDialog.getExistingDirectory(self.centralwidget, "选取文件夹", "C:/")
        if not self.data_path:
            return
        self.mix_list = []
        self.mix_data['x'] = []
        self.mix_data['y'] = []
        self.mix_data['it'] = []
        datas = self.read_spectra(self.data_path)
        for data in datas:
            self.mix_data['x'].append(np.array(json.loads(data[3])))
            self.mix_data['y'].append(np.array(json.loads(data[2])))
            self.mix_data['it'].append(data[4])
            self.mix_list.append(data[1])
        if not self.mix_list:
            QMessageBox.information(self, "Information", 'No available spectral data were found')
        data_list_model = QStringListModel()
        data_list_model.setStringList(self.mix_list)
        self.data_list.setModel(data_list_model)
        self.tabWidget.setCurrentIndex(1)

    def data_display_menu(self, pos):
        menu = QMenu()
        delete_group = menu.addAction("Delete Group")
        change_group_name = menu.addAction("Change Group Name")
        link_Models = menu.addAction("Link Models")
        menu.addSeparator()
        add_spectra = menu.addAction("Add Spectra")
        delete_spectra = menu.addAction("Delete Spectra")
        item = self.data_display.currentItem()
        if not item:
            delete_group.setEnabled(False)
            change_group_name.setEnabled(False)
            link_Models.setEnabled(False)
            add_spectra.setEnabled(False)
            delete_spectra.setEnabled(False)
        else:
            if item.childCount():
                delete_group.setEnabled(True)
                change_group_name.setEnabled(True)
                link_Models.setEnabled(True)
                add_spectra.setEnabled(True)
                delete_spectra.setEnabled(False)
            else:
                delete_group.setEnabled(False)
                change_group_name.setEnabled(False)
                link_Models.setEnabled(False)
                add_spectra.setEnabled(False)
                delete_spectra.setEnabled(True)
        action = menu.exec_(self.data_display.mapToGlobal(pos))
        if action == delete_group:
            AppWindow.delete_group(self)
        elif action == change_group_name:
            AppWindow.change_group_name(self, item)
        elif action == link_Models:
            AppWindow.link_Models(self, item)
        elif action == add_spectra:
            AppWindow.add_spectra(self, item)
        elif action == delete_spectra:
            AppWindow.delete_spectra(self)

    def read_spectra(self, spectra_path):
        datas = []
        pathDir = os.listdir(spectra_path)
        for s in pathDir:
            newfile = os.path.join(spectra_path, s)
            if os.path.isfile(newfile):
                if os.path.splitext(newfile)[1] == ".txt":
                    try:
                        readfile = parseBWTekFile(newfile, s, select_ramanshift=False, xname='Raman Shift',
                                                  yname="Dark Subtracted #1")
                        raw_axis = readfile['axis']
                        raw_spectrum = readfile['spectrum']
                        raw_axis, raw_spectrum = zip(*sorted(zip(raw_axis, raw_spectrum)))
                        name = readfile['name']
                        inter_time = readfile['integral_time']
                        datas.append([None, name, json.dumps(list(raw_spectrum)),
                                      json.dumps(list(raw_axis)), inter_time, 0])
                    except:
                        raw_axis, raw_spectrum = self.read_simple_txt(newfile)
                        raw_axis, raw_spectrum = zip(*sorted(zip(raw_axis, raw_spectrum)))
                        name = os.path.splitext(s)[0]
                        datas.append([None, name, json.dumps(list(raw_spectrum)), json.dumps(list(raw_axis)), 0.0, 0])

                elif os.path.splitext(newfile)[1].lower() == ".spc":
                    axis, x, y = readSPC(newfile)
                    raw_axis = x.reshape(-1)
                    raw_spectrum = y.reshape(-1)
                    raw_axis, raw_spectrum = zip(*sorted(zip(raw_axis, raw_spectrum)))
                    name = os.path.splitext(s)[0]
                    datas.append([None, name, json.dumps(list(raw_spectrum)), json.dumps(list(raw_axis)), 0.0, 0])

                elif os.path.splitext(newfile)[1].lower() == ".jdx":
                    readfile = jcamp.JCAMP_reader(newfile)
                    raw_axis = readfile['x']
                    raw_spectrum = readfile['y']
                    raw_axis, raw_spectrum = zip(*sorted(zip(raw_axis, raw_spectrum)))
                    name = readfile['title']
                    datas.append([None, name, json.dumps(list(raw_spectrum)), json.dumps(list(raw_axis)), 0.0, 0])

                elif os.path.splitext(newfile)[1].lower() == ".db":
                    with sqlite3.connect(newfile) as con:
                        con.row_factory = sqlite3.Row
                        query_str = 'SELECT * FROM standardSamples where include=1'
                        rows = con.cursor().execute(query_str).fetchall()
                        for i, row in enumerate(rows):
                            spec = {}
                            for key in row.keys():
                                spec[key] = row[key]
                            name = spec['name']
                            inter_time = spec['integral']
                            raw_spectrum = np.frombuffer(spec['spectrum'], dtype=np.float32)
                            raw_axis = np.linspace(spec['XStart'], spec['XEnd'],
                                                   int((spec['XEnd'] - spec['XStart']) / spec['XInterval']) + 1)
                            datas.append([None, name, json.dumps(list(raw_spectrum)),
                                          json.dumps(list(raw_axis)), inter_time, 0])
        return datas

    def load_spectra(self):
        if not self.db:
            return
        spectra_path = QFileDialog.getExistingDirectory(self.centralwidget, "Choose data path", "C:/")
        if not spectra_path:
            return
        datas = AppWindow.read_spectra(self, spectra_path)
        if not datas:
            QMessageBox.information(self, "Information", 'No available spectral data were found')
            return
        sql = "select Group_Name from Groups"
        self.cur.execute(sql)
        group_db = self.cur.fetchall()
        name_list = [line[0] for line in group_db]
        num = 0
        group_name = 'Models' + str(num)
        while group_name in name_list:
            num += 1
            group_name = 'Models' + str(num)
        sql = 'insert into Groups VALUES (?,?)'
        self.cur.execute(sql, (None, group_name))
        self.db.commit()
        sql = 'select Group_ID from Groups where Group_Name=?'
        self.cur.execute(sql, (group_name,))
        group_id = self.cur.fetchone()[0]
        for i in range(len(datas)):
            datas[i].append(group_id)
        sql = 'insert into Component_Info VALUES (?,?,?,?,?,?,?)'
        self.cur.executemany(sql, datas)
        self.db.commit()
        root = QTreeWidgetItem(self.data_display)
        root.setText(0, group_name)
        for column in datas:
            child = QTreeWidgetItem()
            child.setText(0, column[1])
            child.setText(1, self.model_ref[column[5]])
            root.addChild(child)

    def add_spectra(self, item):
        if not item.childCount():
            return
        spectra_path = QFileDialog.getExistingDirectory(self.centralwidget, "选取文件夹", "C:/")
        if not spectra_path:
            return
        datas = AppWindow.read_spectra(self, spectra_path)
        if not datas:
            QMessageBox.information(self, "Information", 'No available spectral data were found')
            return
        sql = 'select Group_ID from Groups where Group_Name=?'
        self.cur.execute(sql, (item.text(0),))
        group_id = self.cur.fetchone()[0]
        for i in range(len(datas)):
            datas[i].append(group_id)
        sql = 'insert into Component_Info VALUES (?,?,?,?,?,?,?)'
        self.cur.executemany(sql, datas)
        self.db.commit()
        for column in datas:
            child = QTreeWidgetItem()
            child.setText(0, column[1])
            child.setText(1, self.model_ref[column[5]])
            item.addChild(child)

    def delete_spectra(self):
        item = self.data_display.currentItem()
        if not item:
            return
        if not item.childCount():
            name = item.text(0)
            group_name = item.parent().text(0)
            reply = QMessageBox.question(self.centralwidget, 'Delete', "Do you want to detele '%s' ?" % name,
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                sql = 'select Group_ID from Groups where Group_Name=?'
                current_id = self.cur.execute(sql, (group_name,)).fetchone()[0]
                sql = 'delete from Component_Info where Component_Name=? and From_Group=?'
                self.cur.execute(sql, (name, current_id))

                self.db.commit()
                item.parent().removeChild(item)
            else:
                return
        else:
            return

    def delete_group(self):
        item = self.data_display.currentItem()
        if not item:
            return
        if item.childCount():
            group_name = item.text(0)
            reply = QMessageBox.question(self.centralwidget, 'Delete', 'Do you want to detele Table "%s" ?' % group_name
                                         , QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
            if reply == QMessageBox.Yes:
                sql = 'delete from Groups where Group_Name=?'
                self.cur.execute(sql, (group_name,))
                self.db.commit()
                index = self.data_display.indexOfTopLevelItem(item)
                self.data_display.takeTopLevelItem(index)
            else:
                return
        else:
            return

    def change_group_name(self, item):
        name = item.text(0)
        childwin = ChangeName(name)
        childwin.signal_parp.connect(self.change_name_signal)  # 主窗口接收信号
        childwin.exec_()

    def change_name_signal(self, m):
        self.cur.execute("select Group_Name from Groups")
        group_names = [line[0] for line in self.cur.fetchall()]
        item = self.data_display.currentItem()
        name = item.text(0)
        if m in group_names:
            if m == name:
                return
            else:
                QMessageBox.information(self, "Information", 'Already have Table called "%s"' % m)
                return
        item.setText(0, m)
        sql = 'select Group_ID from Groups where Group_Name=?'
        current_id = self.cur.execute(sql, (name,)).fetchone()[0]
        sql = 'update Groups set Group_Name=? where Group_ID=?'
        self.cur.execute(sql, (m, current_id))
        self.db.commit()

    def link_Models(self, item):
        self.link_widget = item
        group = item.text(0)
        sql = 'select Group_ID from Groups where Group_Name=?'
        group_id = self.cur.execute(sql, (group,)).fetchone()[0]
        print(group_id)
        sql = 'select * from Group_Model_Info where From_Group=?'
        old_para = self.cur.execute(sql, (group_id,)).fetchone()
        if not old_para:
            old_para = [group_id]
        childwin = LinkModels(old_para=old_para)
        childwin.move(self.geometry().x() + (self.geometry().width() - childwin.width()) // 2,
                      self.geometry().y() + (self.geometry().height() - childwin.height()) // 2)
        childwin.signal_parp.connect(self.get_link_signal)
        childwin.exec_()

    def get_link_signal(self, m):
        group_id = m[0]
        correct_models = m[-1]
        sql = 'select Component_Name from Component_Info where From_Group=?'
        names_db = self.cur.execute(sql, (group_id,)).fetchall()
        names = [n[0] for n in names_db]
        sql = 'update Component_Info set Model=? where Component_Name=? and From_Group=?'
        for name in names:
            if name in correct_models:
                self.cur.execute(sql, (1, name, group_id))
                self.link_widget.child(names.index(name)).setText(1, self.model_ref[1])
            else:
                self.cur.execute(sql, (0, name, group_id))
                self.link_widget.child(names.index(name)).setText(1, self.model_ref[0])
        sql = 'select * from Group_Model_Info where From_Group=?'
        old_para = self.cur.execute(sql, (group_id,)).fetchone()
        if not old_para:
            sql = 'insert into Group_Model_Info VALUES (?,?,?,?,?,?)'
            self.cur.execute(sql, (m[1], m[2], m[3], '', m[4], group_id))
        else:
            sql = 'update Group_Model_Info set Raman_Start=?, Raman_End=?, Raman_Interval=?, Save_Path=? where ' \
                  'From_Group=? '
            self.cur.execute(sql, (m[1], m[2], m[3], m[4], group_id))
        self.db.commit()

    def click_to_plot(self):
        if self.plot_lock:
            QMessageBox.information(self, "Information", 'Please wait until result save process finished')
            return
        item = self.data_display.currentItem()
        if item.childCount():
            return
        name = item.text(0)
        group = item.parent().text(0)
        sql = 'select Group_ID from Groups where Group_Name=?'
        group_id = self.cur.execute(sql, (group,)).fetchone()[0]
        sql = 'select Raw_Spectrum,Raw_Axis from Component_Info where Component_Name=? and From_Group=?'
        datas = self.cur.execute(sql, (name, group_id)).fetchall()[0]
        data_array = np.array(json.loads(datas[0]))
        # data_array = data_array / np.max(data_array)
        shift_array = np.array(json.loads(datas[1]))
        if not self.muti:
            self.fig_2.axes.cla()
        self.fig_2.axes.plot(shift_array, data_array, label=name)
        self.fig_2.axes.set_xlabel("Raman Shift", fontsize=12, color='k')
        self.fig_2.axes.set_ylabel("intensity", fontsize=12, color='k')
        self.fig_2.axes.legend(loc='best')
        self.fig_2.draw()

    def click_to_plot_mix(self):
        if self.plot_lock:
            QMessageBox.information(self, "Information", 'Please wait until result save process finished')
            return
        idx = self.data_list.currentIndex().row()
        name = self.mix_list[idx]
        data_array = self.mix_data['y'][idx]
        shift_array = self.mix_data['x'][idx]
        if not self.muti:
            self.fig_2.axes.cla()
        self.fig_2.axes.plot(shift_array, data_array, label=name)
        self.fig_2.axes.set_xlabel("Raman Shift", fontsize=12, color='k')
        self.fig_2.axes.set_ylabel("intensity", fontsize=12, color='k')
        self.fig_2.axes.legend(loc='best')
        self.fig_2.draw()

    def mutiplot_func(self):
        if self.muti:
            self.muti = False
        else:
            self.muti = True

    def erase_plot_func(self):
        self.fig_2.axes.cla()
        self.fig_2.axes.set_xlabel("Raman Shift", fontsize=12, color='k')
        self.fig_2.axes.set_ylabel("intensity", fontsize=12, color='k')
        self.fig_2.draw()

    def train_models(self):
        if self.train_on:
            QMessageBox.information(self, "Information", 'A training process is running')
            return
        item = self.data_display.currentItem()
        if not item:
            return
        self.train_index = []
        self.train_com_name = []
        self.train_com_spec = []
        self.t_axis = []
        GroupMI = None
        ComponentMI = None
        fixed = False
        if not item.childCount():
            self.train_group_widget = item.parent()
            self.train_group = item.parent().text(0)
            sql = 'select Group_ID from Groups where Group_Name=?'
            group_id = self.cur.execute(sql, (self.train_group,)).fetchone()[0]
            sql = 'select Component_Name from Component_Info where From_Group=?'
            names = self.cur.execute(sql, (group_id,)).fetchall()
            for name in names:
                self.train_com_name.append(name[0])
            if item.text(2) == 'Yes':
                reply = QMessageBox.question(self, 'Train', 'Do you want to re-train "%s"?' % item.text(0),
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if reply == QMessageBox.No:
                    return
                elif reply == QMessageBox.Yes:
                    self.train_index = [self.train_com_name.index(item.text(0))]
                else:
                    return
            else:
                self.train_index = [self.train_com_name.index(item.text(0))]
                sql = 'select Component_ID from Component_Info where Component_Name=? and From_Group=?'
                index = self.cur.execute(sql, (item.text(0), group_id)).fetchone()[0]
                try:
                    sql = 'select * from Component_Model_Info where From_Component=?'
                    ComponentMI = self.cur.execute(sql, (index,)).fetchone()
                except:
                    ComponentMI = None
                sql = 'select Component_ID from Component_Info where Model=1 and From_Group=?'
                jug = self.cur.execute(sql, (group_id,)).fetchall()
                if len(jug) >= 2:
                    fixed = True
        else:
            self.train_group_widget = item
            self.train_group = item.text(0)
            sql = 'select Group_ID from Groups where Group_Name=?'
            group_id = self.cur.execute(sql, (self.train_group,)).fetchone()[0]
            sql = 'select Component_Name from Component_Info where From_Group=?'
            names = self.cur.execute(sql, (group_id,)).fetchall()
            for name in names:
                self.train_com_name.append(name[0])
            sql = 'select Component_Name from Component_Info where Model=1 and From_Group=?'
            jug = self.cur.execute(sql, (group_id,)).fetchone()
            if jug:
                reply = QMessageBox.question(self, 'Train', "Already have some trained models \n "
                                                            "Do you want to re-train them?",
                                             QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
                if reply == QMessageBox.Yes:
                    self.train_index = list(np.arange(0, len(self.train_com_name)))
                elif reply == QMessageBox.No:
                    sql = 'select Component_Name from Component_Info where Model=0 and From_Group=?'
                    components = self.cur.execute(sql, (group_id,)).fetchall()
                    if not components:
                        return
                    for com in components:
                        self.train_index.append(self.train_com_name.index(com[0]))
                    fixed = True
                else:
                    return
            else:
                self.train_index = list(np.arange(0, len(self.train_com_name)))
        sql = 'select Raw_Spectrum, Raw_Axis from Component_Info where From_Group=?'
        datas = self.cur.execute(sql, (group_id,)).fetchall()
        for data in datas:
            self.train_com_spec.append(np.array(json.loads(data[0])))
            self.t_axis.append(np.array(json.loads(data[1])))
        sql = 'select * from Group_Model_Info where From_Group=?'
        old_para = self.cur.execute(sql, (group_id,)).fetchone()
        if old_para is not None:
            GroupMI = old_para
        self.create_models(GroupMI=GroupMI, ComponentMI=ComponentMI, fixed=fixed)

    def create_models(self, GroupMI=None, ComponentMI=None, fixed=False):
        childwin = TrainNewP(GroupMI=GroupMI, ComponentMI=ComponentMI, fixed=fixed)
        childwin.move(self.geometry().x() + (self.geometry().width() - childwin.width()) // 2,
                      self.geometry().y() + (self.geometry().height() - childwin.height()) // 2)
        childwin.signal_parp.connect(self.get_train_signal)  # 主窗口接收信号
        childwin.exec_()

    def get_train_signal(self, m):
        self.GroupMI = [m[0], m[1], m[2], m[9], m[10]]
        self.ComponentMI = [m[7], m[8], m[3], m[4], m[5], m[6]]
        self.model_path = m[-1]
        self.t_new_axis = np.linspace(m[0], m[1], int((m[1] - m[0]) / m[2] + 1))
        AppWindow.train_run(self, m[3:], self.train_index, self.t_axis, self.t_new_axis)

    def train_run(self, sp, count, axis, new_axis):
        optimizer_list = [tf.keras.optimizers.Adam(lr=sp[1]), tf.keras.optimizers.Adadelta(lr=sp[1]),
                          tf.keras.optimizers.Adagrad(lr=sp[1]), tf.keras.optimizers.Adamax(lr=sp[1])]
        optimizer = optimizer_list[sp[0]]
        train_para = [optimizer, sp[2], sp[3], sp[-1]]
        aug_para = sp[4:7]
        info_para = [self.train_com_name, count, self.train_com_spec, axis, new_axis]

        self.thread = TrainRun(train_para, aug_para, info_para)
        self.thread.signal.connect(self.get_train_thread_signal)
        self.thread.process_signal.connect(self.get_train_process_signal)
        self.thread.err_signal.connect(self.get_train_err_signal)
        self.thread.data_signal.connect(self.get_train_data_signal)
        self.thread.para_signal.connect(self.get_train_para_signal)
        self.thread.daemon = True
        self.thread.start()

    def get_train_thread_signal(self, m):
        if m == 'run':
            self.progressBar.setMinimum(0)
            self.progressBar.setMaximum(0)
            self.textBrowser_3.setText('Training CNN models')
            self.train_on = True
            self.train_para = {}
        elif m == 'finished':
            self.progressBar.setMinimum(0)
            self.progressBar.setMaximum(100)
            self.textBrowser_3.setText('Finished')
            self.train_on = False
            QtCore.QTimer().singleShot(2000, self.clear_text_1)
            childwin = TrainReport(self.train_para)
            childwin.exec_()
        else:
            QMessageBox.information(self, "Information", m)

    def get_train_process_signal(self, m):
        self.textBrowser_3.setText(m)

    def get_train_err_signal(self, m):
        QMessageBox.information(self, "Information", m)
        self.train_on = False
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(100)
        self.textBrowser_3.setText('')

    def get_train_data_signal(self, m):
        group = self.train_group
        name = m[0]
        sql = 'select Group_ID from Groups where Group_Name=?'
        group_id = self.cur.execute(sql, (group,)).fetchone()[0]
        sql = 'update Component_Info set Model=? where Component_Name=? and From_Group=?'
        self.cur.execute(sql, (1, name, group_id))
        GroupMI = self.GroupMI.copy()
        GroupMI.append(group_id)
        sql = 'select * from Group_Model_Info where From_Group=?'
        jug = self.cur.execute(sql, (group_id,)).fetchone()
        if jug:
            sql = 'update Group_Model_Info set Raman_Start=?, Raman_End=?, Raman_Interval=?, Aug_Save_Path=?, ' \
                  'Save_Path=? where From_Group=? '
            self.cur.execute(sql, tuple(GroupMI))
        else:
            sql = 'insert into Group_Model_Info values (?,?,?,?,?,?)'
            self.cur.execute(sql, GroupMI)
        sql = 'select Component_ID from Component_Info where Component_Name=?'
        component_id = self.cur.execute(sql, (name,)).fetchone()[0]
        ComponentMI = self.ComponentMI.copy()
        ComponentMI.append(component_id)
        sql = 'select * from Component_Model_Info where From_Component=?'
        jug = self.cur.execute(sql, (component_id,)).fetchone()
        if jug:
            sql = 'update Component_Model_Info set Augment_Num=?, Noise_Rate=?, Optimizer=?, LR=?, BS=?, EPS=? where ' \
                  'From_Component=? '
            self.cur.execute(sql, tuple(ComponentMI))
        else:
            sql = 'insert into Component_Model_Info values (?,?,?,?,?,?,?)'
            self.cur.execute(sql, ComponentMI)
        self.db.commit()
        self.train_group_widget.child(self.train_com_name.index(name)).setText(1, self.model_ref[1])

    def get_train_para_signal(self, m):
        self.train_para[m[0]] = [m[1], m[2], m[3], m[4], m[5], m[6]]

    def selected_items(self, data_list, names):
        selected = data_list.selectedIndexes()
        data_ = []
        index = []
        length = len(selected)
        for i in range(length):
            num = selected[i].row()
            data = list(names)[num]
            data_.append(data)
            index.append(num)
        return data_, index

    def get_pred_data(self, shift_para):
        self.pred_names, index = AppWindow.selected_items(self, self.data_list, self.mix_list)
        if not self.pred_names:
            return None
        data_array = np.zeros((1, len(shift_para)))
        for i in index:
            raw_axis = self.mix_data['x'][i]
            raw_spectrum = self.mix_data['y'][i]
            spectrum = np.interp(shift_para, raw_axis, raw_spectrum).astype(np.float64).copy()
            data_array = np.vstack((data_array, spectrum.T))
        data_array = np.delete(data_array, 0, 0)
        return data_array

    def predict_process_func(self):
        if self.pred_on:
            QMessageBox.information(self, "Information", 'A prediction process is running')
            return
        if not self.db:
            return
        sql = 'select Group_Name from Groups'
        self.cur.execute(sql)
        tab_name_db = self.cur.fetchall()
        tab_names = [line[0] for line in tab_name_db]
        childwin = PredictionSetting(tab_names)
        childwin.signal_parp.connect(self.predict_process)
        childwin.exec_()

    def predict_process(self, m):
        if not m:
            return
        self.pred_group = m[0]
        self.threshold = m[1]
        sql = 'select Group_ID from Groups where Group_Name=?'
        group_id = self.cur.execute(sql, (self.pred_group,)).fetchone()[0]
        sql = 'select * from Group_Model_Info where From_Group=?'
        group_para = self.cur.execute(sql, (group_id,)).fetchone()
        if not group_para:
            QMessageBox.information(self, "Information", 'No available models')
            return
        self.axis = np.arange(group_para[0], group_para[1], group_para[2])
        model_path_list = []
        dir = os.listdir(group_para[-2])
        for file in dir:
            model_path_list.append(os.path.join(group_para[-2], file))
        self.pred_data = self.get_pred_data(self.axis)
        if self.pred_data is None:
            QMessageBox.information(self, "Information", 'Missing spectra to be analyzed')
            return
        self.thread_2 = PredRun(self.pred_data, model_path_list)
        self.thread_2.signal.connect(self.get_pred_thread_signal)
        self.thread_2.data_signal.connect(self.get_pred_data_signal)
        self.thread_2.daemon = True
        self.thread_2.start()

    def get_pred_thread_signal(self, m):
        if m == 'run':
            self.progressBar_2.setMinimum(0)
            self.progressBar_2.setMaximum(0)
            if self.QA_on:
                self.textBrowser_4.setText('Quantitative analysis')
            else:
                self.textBrowser_4.setText('Prediction')
            self.pred_on = True
        elif m == 'finished':
            self.progressBar_2.setMinimum(0)
            self.progressBar_2.setMaximum(100)
            self.textBrowser_4.setText('Finished')
            self.pred_on = False
            self.qa.setEnabled(True)
            QtCore.QTimer().singleShot(2000, self.clear_text_2)
        else:
            QMessageBox.information(self, "Information", m)
            self.progressBar_2.setMinimum(0)
            self.progressBar_2.setMaximum(100)
            self.textBrowser_4.setText('')
            self.pred_on = False
            self.QA_on = False

    def get_pred_data_signal(self, m):
        self.pred_prob = np.asarray(m)
        self.result_list = {}
        sql = 'select Group_ID from Groups where Group_Name=?'
        group_id = self.cur.execute(sql, (self.pred_group,)).fetchone()[0]
        sql = 'select Component_Name from Component_Info where Model=1 and From_Group=?'
        component_list = self.cur.execute(sql, (group_id,)).fetchall()
        for i in range(len(self.pred_names)):
            num = np.where(self.pred_prob[:, i] >= self.threshold)[0]
            self.result_list[self.pred_names[i]] = [component_list[j][0] for j in num]
        self.predict_result.clear()
        self.main_root = QTreeWidgetItem(self.predict_result)
        self.main_root.setText(0, 'results')
        for i in range(len(self.pred_names)):
            root = QTreeWidgetItem()
            root.setText(0, self.pred_names[i])
            root.setCheckState(0, Qt.Unchecked)
            self.main_root.addChild(root)
            for j in range(len(self.result_list[self.pred_names[i]])):
                child = QTreeWidgetItem()
                child.setText(0, self.result_list[self.pred_names[i]][j])
                root.addChild(child)
        self.predict_result.addTopLevelItem(self.main_root)
        self.predict_result.expandAll()
        self.tabWidget.setCurrentIndex(2)

    def QA_process_func(self):
        if self.QA_on:
            QMessageBox.information(self, "Information", 'A prediction process is running')
            return
        if not self.db:
            return
        childwin = RatioEstimation()
        childwin.signal_parp.connect(self.quantitative_analysis)
        childwin.exec_()

    def quantitative_analysis(self, m):
        self.QA_on = True
        sql = 'select Group_ID from Groups where Group_Name=?'
        group_id = self.cur.execute(sql, (self.pred_group,)).fetchone()[0]
        i = 0
        mix = []
        com = []
        for pn in self.pred_names:
            x_size = self.axis.shape[0]
            Spectrum_data = np.zeros((1, x_size))
            preds = self.result_list[pn]
            for pred in preds:
                sql = 'select Raw_Axis, Raw_Spectrum, Inter_time from Component_Info where Component_Name=? and ' \
                      'From_Group=? '
                spectra_info = self.cur.execute(sql, (pred, group_id)).fetchone()
                old_axis = np.array(json.loads(spectra_info[0]))
                old_spectrum = np.array(json.loads(spectra_info[1]))
                spectrum = np.interp(self.axis, old_axis, old_spectrum).astype(np.float64)
                try:
                    inter_time = spectra_info[2]
                    spectrum = spectrum / inter_time
                except:
                    pass
                Spectrum_data = np.vstack((Spectrum_data, spectrum.T))
            spectra_raw = np.delete(Spectrum_data, 0, 0)
            if self.mix_data['it']:
                mixture = self.pred_data[i] / self.mix_data['it'][i]
            else:
                mixture = self.pred_data[i]
            mix.append(mixture)
            com.append(spectra_raw)
            i += 1
        self.thread_3 = QARun(mix, com, m)
        self.thread_3.signal.connect(self.get_pred_thread_signal)
        self.thread_3.rate_signal.connect(self.qa_display)
        self.thread_3.daemon = True
        self.thread_3.start()

    def qa_display(self, ratios):
        self.ratios = ratios
        self.predict_result.clear()
        self.main_root = QTreeWidgetItem(self.predict_result)
        self.main_root.setText(0, 'Prediction results')
        for i in range(len(self.pred_names)):
            root = QTreeWidgetItem()
            root.setText(0, self.pred_names[i])
            root.setCheckState(0, Qt.Unchecked)
            self.main_root.addChild(root)
            for j in range(len(self.result_list[self.pred_names[i]])):
                child = QTreeWidgetItem()
                child.setText(0, self.result_list[self.pred_names[i]][j])
                child.setText(1, str(self.ratios[i][j]))
                root.addChild(child)
        self.predict_result.addTopLevelItem(self.main_root)
        self.predict_result.expandAll()
        self.tabWidget.setCurrentIndex(2)
        self.QA_on = False

    def clear_text_1(self):
        self.textBrowser_3.setText('')

    def clear_text_2(self):
        self.textBrowser_4.setText('')

    def get_checked(self, item):
        if self.plot_lock:
            QMessageBox.information(self, "Information", 'Please wait until saving process finished')
            return
        pred = ''
        component_list = []
        if not item:
            return
        if item.checkState(0) == Qt.Checked:
            pred = item.text(0)
            for j in range(item.childCount()):
                component_list.append(item.child(j).text(0))
            for i in range(self.main_root.childCount()):
                if self.main_root.child(i) == item:
                    continue
                else:
                    self.main_root.child(i).setCheckState(0, Qt.Unchecked)
        self.fig.axes.cla()
        self.fig.axes.set_xlabel("Raman Shift", fontsize=12, color='k')
        self.fig.axes.set_ylabel("Intensity", fontsize=12, color='k')
        if pred:
            sql = 'select Group_ID from Groups where Group_Name=?'
            group_id = self.cur.execute(sql, (self.pred_group,)).fetchone()[0]
            pred_idx = self.pred_names.index(pred)
            mix_data = self.pred_data[pred_idx]
            self.fig.axes.plot(self.axis, mix_data, label=pred)
            if component_list:
                for m in range(len(component_list)):
                    sql = 'select Raw_Axis, Raw_Spectrum, Inter_time from Component_Info where Component_Name=? and ' \
                          'From_Group=? '
                    spectra_info = self.cur.execute(sql, (component_list[m], group_id)).fetchone()
                    old_axis = np.array(json.loads(spectra_info[0]))
                    old_spectrum = np.array(json.loads(spectra_info[1]))
                    spectrum = np.interp(self.axis, old_axis, old_spectrum).astype(np.float64)
                    if self.ratios:
                        spectrum = spectrum * self.ratios[pred_idx][m]
                    self.fig.axes.plot(self.axis, spectrum, label=component_list[m])
                self.fig.axes.legend(loc='best')
        self.fig.draw()

    def save_function(self):
        if not self.result_list:
            return
        save_path, ext = QFileDialog.getSaveFileName(self.centralwidget, "Choose result save path", "C:/",
                                                     "EXCEL(*.xlsx)")
        if not save_path:
            return
        self.thread_s = EXCELCreate(self.pred_names, self.result_list, self.ratios, save_path)
        self.thread_s.signal.connect(self.get_save_thread_signal)
        self.thread_s.daemon = True
        self.thread_s.start()

    def get_save_thread_signal(self, m):
        if m == 'run':
            self.progressBar_2.setMinimum(0)
            self.progressBar_2.setMaximum(0)
            self.textBrowser_4.setText('Saving results')
            self.plot_lock = True
        elif m == 'finish':
            self.progressBar_2.setMaximum(100)
            self.textBrowser_4.setText('Finished')
            self.plot_lock = False
            QtCore.QTimer().singleShot(2000, self.clear_text_2)
        else:
            QMessageBox.information(self, "Information", m)
            self.progressBar_2.setMaximum(100)
            self.textBrowser_4.setText('')
            self.plot_lock = False

    def closeEvent(self, event):
        messageBox = QMessageBox(QMessageBox.Question, "Confirm Exit", "Are you sure you want to exit EasyCID?")
        Qyes = messageBox.addButton(self.tr("Exit"), QMessageBox.YesRole)
        Qno = messageBox.addButton(self.tr("Cancel"), QMessageBox.NoRole)
        messageBox.exec_()
        if messageBox.clickedButton() == Qyes:
            event.accept()
        elif messageBox.clickedButton() == Qno:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = AppWindow()
    win.show()
    sys.exit(app.exec_())
