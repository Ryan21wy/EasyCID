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
import inspect
import ctypes
import chardet

matplotlib.use('Agg')

import matplotlib.pyplot as plt
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
from child_win import QA_win
from child_win import TrainHistory_win


def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


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


def noise(spectrum, nr):
    N_MAX = nr * np.max(spectrum)
    Xnoise = np.random.normal(0, N_MAX, (1, spectrum.shape[1]))
    spectrum_noise = spectrum + Xnoise
    return np.maximum(spectrum_noise, 0)


def component_in(spectrum_raw, component, num, nr=0):
    c1 = np.random.uniform(0.1, 1, (num, 1))
    c2 = np.random.rand(num, 1)
    k = np.random.randint(0, spectrum_raw.shape[0], size=(num, 1))

    a = c2 / c1

    Spectrumdata_new = np.zeros((1, spectrum_raw.shape[1]))
    component_in = (spectrum_raw[component, :]).reshape((1, spectrum_raw.shape[1]))

    for i in range(num):
        Spectrumdata_new2 = c1[i] * component_in + c2[i] * spectrum_raw[k[i], :]
        Spectrumdata_new2 = noise(Spectrumdata_new2, nr)
        Spectrumdata_new = np.vstack((Spectrumdata_new, Spectrumdata_new2))

    Spectrumdata_new = np.delete(Spectrumdata_new, 0, 0)
    label = np.ones((num, 1))
    return {'spectrum_in': Spectrumdata_new, 'label_in': label}


def component_out(spectrum_raw, component, num, nr=0):
    c1 = np.random.rand(num, 1)
    c2 = np.random.rand(num, 1)
    a = c2 / c1
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
        Spectrumdata_new2 = noise(Spectrumdata_new2, nr)
        Spectrumdata_new = np.vstack((Spectrumdata_new, Spectrumdata_new2))
    Spectrumdata_new = np.delete(Spectrumdata_new, 0, 0)
    label = np.zeros((num, 1))
    return {'spectrum_out': Spectrumdata_new, 'label_out': label}


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


class TrainNewP(QDialog):
    signal_parp = pyqtSignal(list)

    def __init__(self, raman_shift=None, model_path=None):
        QDialog.__init__(self)
        self.child = TrainNewP_win.Ui_Dialog()
        self.child.setupUi(self)
        self.child.dir_choose.setIcon(QIcon('./EasyCID_Icon/dir.png'))
        self.child.dir_choose.clicked.connect(self.dir_chose)
        self.child.save_.clicked.connect(self.signal_emit)
        self.child.reset_.clicked.connect(self.value_reset)
        self.child.save_.setEnabled(True)
        self.signal = []
        self.raman_shift = None
        self.model_path = None
        if raman_shift:
            self.raman_shift = raman_shift
            self.child.startshift.setValue(raman_shift[0])
            self.child.startshift.setEnabled(False)
            self.child.endshift.setValue(raman_shift[1])
            self.child.endshift.setEnabled(False)
            self.child.interval.setValue(raman_shift[2])
            self.child.interval.setEnabled(False)
        if model_path:
            self.child.savepath.setText(model_path)
            self.child.savepath.setEnabled(False)
            self.child.dir_choose.setEnabled(False)

    def dir_chose(self):
        save_path = QFileDialog.getExistingDirectory(win.centralwidget, "choose save path", "C:/")
        self.child.savepath.setText(save_path)

    def signal_emit(self):
        self.signal = []
        self.signal.append(self.child.startshift.value())
        self.signal.append(self.child.endshift.value())
        self.signal.append(self.child.interval.value())
        self.signal.append(self.child.optimizer_.currentText())
        if self.child.learnrate.text().replace(".", '').isdigit():
            self.signal.append(float(self.child.learnrate.text()))
        else:
            QMessageBox.warning(self, "error", 'The learning rate is unreadable\nPlease enter a reasonable value!')
            return
        self.signal.append(self.child.batchsize.value())
        self.signal.append(self.child.epochs_.value())
        self.signal.append(self.child.number.value())
        if self.child.noise.text().replace(".", '').isdigit():
            self.signal.append(float(self.child.noise.text()))
        else:
            QMessageBox.warning(self, "error", 'The noise rate is unreadable\nPlease enter a reasonable value!')
            return
        if not self.child.savepath.text():
            QMessageBox.warning(self, "error", 'The save path cannot be empty!')
            return
        self.signal.append(self.child.savepath.text())
        self.signal_parp.emit(self.signal)
        TrainNewP.close(self)

    def value_reset(self):
        if self.raman_shift is None:
            self.child.startshift.setValue(240)
            self.child.endshift.setValue(2000)
            self.child.interval.setValue(2)
        self.child.optimizer_.setCurrentIndex(0)
        self.child.learnrate.setText("0.00001")
        self.child.batchsize.setValue(512)
        self.child.epochs_.setValue(500)
        self.child.noise.setText("0.005")
        self.child.number.setValue(30000)
        self.child.save_.setEnabled(False)


class TrainHistory(QDialog):
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
        self.ax1.set_xlabel("Epoch", fontsize=12, color='k')
        self.ax1.set_ylabel("Loss", fontsize=12, color='k')
        self.ax2 = self.ax1.twinx()
        self.ax2.set_ylabel("Accuracy", fontsize=12, color='k')
        self.gridlayout = QGridLayout(self.child.groupBox)
        self.gridlayout.addWidget(self.fig)
        self.gridlayout.addWidget(self.fig_ntb)
        TrainHistory.load(self)

    def load(self):
        self.model = QStandardItemModel(len(self.names), 3)
        self.model.setHorizontalHeaderLabels(['Component', 'test-Loss', 'test-Acc'])
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
        lns1 = self.ax1.plot(x_axe, t_loss, label='train loss', c='b')
        lns2 = self.ax1.plot(x_axe, v_loss, label='validation loss', c='g')
        lns3 = self.ax2.plot(x_axe, t_acc, label='train acc', c='y')
        lns4 = self.ax2.plot(x_axe, v_acc, label='validation acc', c='r')
        lns = lns1 + lns2 + lns3 + lns4
        labs = [l.get_label() for l in lns]
        self.ax1.legend(lns, labs, loc=7)
        self.ax1.set_title(name, fontsize=12, color='b')
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


class QuantitativeAnalysis(QDialog):
    signal_parp = pyqtSignal(dict)

    def __init__(self):
        QDialog.__init__(self)
        self.child = QA_win.Ui_dialog()
        self.child.setupUi(self)

        self.widget_1(False)
        self.widget_2(False)

        self.child.OK.clicked.connect(self.signal_emit)
        self.child.checkBox.stateChanged.connect(self.checkBox_state)
        self.child.checkBox_2.stateChanged.connect(self.checkBox_2_state)

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
        QuantitativeAnalysis.close(self)


class PredictionRun(QDialog):
    signal_parp = pyqtSignal(str)

    def __init__(self, table_list):
        QDialog.__init__(self)
        self.child = ModelSelect_win.Ui_dialog()
        self.child.setupUi(self)
        for item in table_list:
            self.child.comboBox.addItem(item)
        self.child.OK.clicked.connect(self.signal_emit)

    def signal_emit(self):
        table = self.child.comboBox.currentText()
        self.signal_parp.emit(table)
        PredictionRun.close(self)


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
                aug_path = u'./augmentation'
                mkdir(aug_path)
                aug_data_path = os.path.join(aug_path, self.names[num] + '.npy')
                aug_label_path = os.path.join(aug_path, self.names[num] + '_label.npy')
                if os.path.isfile(aug_data_path):
                    spectrum = np.load(aug_data_path)
                    label = np.load(aug_label_path)
                else:
                    num_sample = self.aug_number
                    data_in = component_in(spectra_raw, num, num=int(num_sample / 2), nr=self.noise_rate)
                    Xin = data_in['spectrum_in']
                    Yin = data_in['label_in']
                    data_out = component_out(spectra_raw, num, num=int(num_sample / 2), nr=self.noise_rate)
                    Xout = data_out['spectrum_out']
                    Yout = data_out['label_out']
                    spectrum = np.concatenate((Xin, Xout), axis=0)
                    label = np.concatenate((Yin, Yout), axis=0)
                    np.save(aug_data_path, spectrum)
                    np.save(aug_label_path, label)
                spectrumdata, labeldata = randomize(spectrum, label)
                savepath = os.path.join(self.model_path, self.names[num])
                mkdir(savepath)
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
                model.save_weights(savepath + '/weight_tf_savedmodel.h5')

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
                reload_model.load_weights(path + '/weight_tf_savedmodel.h5')
                Xtest_pre = Xtest_pre.reshape(Xtest_pre.shape[0], Xtest_pre.shape[1], 1)
                y = reload_model.predict(Xtest_pre)
                y_DeepCID.append(y)
            self.data_signal.emit(y_DeepCID)
            self.signal.emit('finished')
        except Exception as err:
            print(str(err))
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
        self.data_display.setColumnCount(3)
        self.data_display.setHeaderLabels(['Component', ' Index ', ' Trained '])
        self.data_display.header().setStretchLastSection(False)
        self.data_display.header().setSectionResizeMode(QHeaderView.Stretch)
        self.data_display.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.data_display.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
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
        self.train_table = ''
        self.pred_table = ''
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
        self.qa.setEnabled(False)
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
            self.cur = self.db.cursor()
            abs_bath = os.path.abspath(path)
            self.modelPath.setTitle('Current Database: "%s"' % abs_bath)
            AppWindow.db_data_display(self)
        except Exception as err:
            QMessageBox.information(self, "Information", str(err))

    def set_up_db(self):
        file_name, type = QFileDialog.getSaveFileName(self.centralwidget, "Build database", "C:/", 'database (*.db)')
        AppWindow.connect_db(self, file_name)

    def link_to_db(self):
        file_name, type = QFileDialog.getOpenFileName(self.centralwidget, "Choose database",
                                                      r"C:/", 'database (*.db)')
        if not file_name:
            return
        AppWindow.connect_db(self, file_name)

    def get_db_data(self, cur):
        cur.execute("select name from sqlite_master where type='table'")
        tab_name_db = cur.fetchall()
        tab_names = [line[0] for line in tab_name_db]
        if not tab_names:
            return None, None
        datas = []
        for tab_name in tab_names:
            data = cur.execute('SELECT Component,Get_trained FROM "%s"' % tab_name)
            datas.append(data.fetchall())

        return tab_names, datas

    def db_data_display(self):
        if not self.db:
            return
        tab_names, datas = AppWindow.get_db_data(self, self.cur)
        if tab_names is None:
            self.data_display.clear()
            return
        self.data_display.clear()
        for i in range(len(tab_names)):
            tab_name = tab_names[i]
            data = datas[i]
            root = QTreeWidgetItem(self.data_display)
            root.setText(0, tab_name)
            j = 0
            for column in data:
                child = QTreeWidgetItem()
                j += 1
                child.setText(1, str(j))
                child.setText(0, column[0])
                child.setText(2, column[1])
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
        except Exception as err:
            print(str(err))
        return np.array(x), np.array(y)

    def open_dir_func(self):
        self.data_path = QFileDialog.getExistingDirectory(self.centralwidget, "选取文件夹", "C:/")
        if not self.data_path:
            return
        dir = os.listdir(self.data_path)
        self.mix_list = []
        self.mix_data['x'] = []
        self.mix_data['y'] = []
        self.mix_data['it'] = []
        for txt in dir:
            data_path = os.path.join(self.data_path, txt)
            try:
                if os.path.splitext(txt)[-1] == '.txt':
                    try:
                        readfile = parseBWTekFile(data_path, txt, select_ramanshift=False,
                                                  xname='Raman Shift', yname="Dark Subtracted #1")
                        readfile['axis'], readfile['spectrum'] = zip(
                            *sorted(zip(readfile['axis'], readfile['spectrum'])))
                        self.mix_data['x'].append(readfile['axis'])
                        self.mix_data['y'].append(readfile['spectrum'])
                        self.mix_data['it'].append(readfile['integral_time'])
                        self.mix_list.append(os.path.splitext(txt)[0])
                    except:
                        x, y = self.read_simple_txt(data_path)
                        x, y = zip(*sorted(zip(x, y)))
                        self.mix_data['x'].append(x)
                        self.mix_data['y'].append(y)
                        self.mix_list.append(os.path.splitext(txt)[0])

                elif os.path.splitext(txt)[-1] == '.csv':
                    x, y = self.read_simple_txt(data_path)
                    x, y = zip(*sorted(zip(x, y)))
                    self.mix_data['x'].append(x)
                    self.mix_data['y'].append(y)
                    self.mix_list.append(os.path.splitext(txt)[0])

                elif os.path.splitext(txt)[-1] == '.spc':
                    axis, x, y = readSPC(data_path)
                    self.mix_data['x'].append(x.reshape(-1))
                    self.mix_data['y'].append(y.reshape(-1))
                    self.mix_list.append(os.path.splitext(txt)[0])
                elif os.path.splitext(txt)[-1] == '.jdx':
                    readfile = jcamp.JCAMP_reader(data_path)
                    self.mix_data['x'].append(readfile['x'])
                    self.mix_data['y'].append(readfile['y'])
                    self.mix_list.append(os.path.splitext(txt)[0])
            except Exception as err:
                QMessageBox.information(self, "Information", str(err))
                continue
        if not self.mix_list:
            QMessageBox.information(self, "Information", 'No available spectral data were found')
        data_list_model = QStringListModel()
        data_list_model.setStringList(self.mix_list)
        self.data_list.setModel(data_list_model)
        self.tabWidget.setCurrentIndex(1)

    def data_display_menu(self, pos):
        menu = QMenu()
        delete_table = menu.addAction("Delete Table")
        change_table_name = menu.addAction("Change Table Name")
        menu.addSeparator()
        add_spectra = menu.addAction("Add Spectra")
        delete_spectra = menu.addAction("Delete Spectra")
        item = self.data_display.currentItem()
        if not item:
            delete_table.setEnabled(False)
            change_table_name.setEnabled(False)
            add_spectra.setEnabled(False)
            delete_spectra.setEnabled(False)
        else:
            if item.childCount():
                delete_table.setEnabled(True)
                change_table_name.setEnabled(True)
                add_spectra.setEnabled(True)
                delete_spectra.setEnabled(False)
            else:
                delete_table.setEnabled(False)
                change_table_name.setEnabled(False)
                add_spectra.setEnabled(False)
                delete_spectra.setEnabled(True)
        action = menu.exec_(self.data_display.mapToGlobal(pos))
        if action == delete_table:
            AppWindow.delete_table(self)
        elif action == change_table_name:
            AppWindow.change_table_name(self, item)
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
                        datas.append((name, 'No', json.dumps(list(raw_spectrum)),
                                      json.dumps(list(raw_axis)), str(inter_time), '', '', ''))
                    except:
                        raw_axis, raw_spectrum = self.read_simple_txt(newfile)
                        raw_axis, raw_spectrum = zip(*sorted(zip(raw_axis, raw_spectrum)))
                        name = os.path.splitext(s)[0]
                        datas.append((name, 'No', json.dumps(list(raw_spectrum)),
                                      json.dumps(list(raw_axis)), '', '', '', ''))

                elif os.path.splitext(newfile)[1].lower() == ".spc":
                    axis, x, y = readSPC(newfile)
                    raw_axis = x.reshape(-1)
                    raw_spectrum = y.reshape(-1)
                    raw_axis, raw_spectrum = zip(*sorted(zip(raw_axis, raw_spectrum)))
                    name = os.path.splitext(s)[0]
                    datas.append((name, 'No', json.dumps(list(raw_spectrum)),
                                  json.dumps(list(raw_axis)), '', '', '', ''))
                elif os.path.splitext(newfile)[1].lower() == ".jdx":
                    readfile = jcamp.JCAMP_reader(newfile)
                    raw_axis = readfile['x']
                    raw_spectrum = readfile['y']
                    raw_axis, raw_spectrum = zip(*sorted(zip(raw_axis, raw_spectrum)))
                    name = readfile['title']
                    datas.append((name, 'No', json.dumps(list(raw_spectrum)),
                                  json.dumps(list(raw_axis)), '', '', '', ''))
                elif os.path.splitext(newfile)[1].lower() == ".db":
                    with sqlite3.connect(newfile) as con:
                        con.row_factory = sqlite3.Row
                        query_str = 'SELECT * FROM standardSamples where include=1'
                        rows = con.cursor().execute(query_str).fetchall()
                        for i, row in enumerate(rows):
                            spec = {}
                            # print row.keys()
                            for key in row.keys():
                                spec[key] = row[key]
                            name = spec['name']
                            spec['integral_time'] = spec['integral']
                            raw_spectrum = np.frombuffer(spec['spectrum'], dtype=np.float32)
                            raw_axis = np.linspace(spec['XStart'], spec['XEnd'],
                                                   int((spec['XEnd'] - spec['XStart']) / spec['XInterval']) + 1)
                            datas.append((name, 'No', json.dumps(raw_spectrum.tolist()),
                                          json.dumps(raw_axis.tolist()), '', '', '', ''))
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
        self.cur.execute("select name from sqlite_master where type='table'")
        tab_name_db = self.cur.fetchall()
        name_list = []
        for name in tab_name_db:
            name_list.append(name[0])
        num = len(tab_name_db)
        table_name = 'Models' + str(num)
        while table_name in name_list:
            num += 1
            table_name = 'Models' + str(num)
        self.cur.execute('CREATE TABLE "%s" (Component TEXT, Get_trained TEXT, Raw_Spectrum TEXT, Raw_Axis TEXT,'
                         'Inter_Time, New_Spectrum TEXT, New_Axis TEXT, Model_Path TEXT)' % table_name)
        add_action = 'INSERT INTO "%s" VALUES (?,?,?,?,?,?,?,?)' % table_name
        self.cur.executemany(add_action, datas)
        self.db.commit()
        root = QTreeWidgetItem(self.data_display)
        root.setText(0, table_name)
        for column in datas:
            child = QTreeWidgetItem()
            child.setText(0, column[0])
            child.setText(1, column[1])
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
        add_action = 'INSERT INTO "%s" VALUES (?,?,?,?,?,?,?,?)' % item.text(0)
        self.cur.executemany(add_action, datas)
        self.db.commit()
        for column in datas:
            child = QTreeWidgetItem()
            child.setText(0, column[0])
            child.setText(1, column[1])
            item.addChild(child)

    def delete_spectra(self):
        item = self.data_display.currentItem()
        if not item:
            return
        if not item.childCount():
            name = item.text(0)
            table_name = item.parent().text(0)
            reply = QMessageBox.question(self.centralwidget, 'Delete', "Do you want to detele '%s' ?" % name,
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                sql = 'DELETE FROM "%s" WHERE Component=?' % table_name
                self.cur.execute(sql, [name])
                self.db.commit()
                item.parent().removeChild(item)
            else:
                return
        else:
            return

    def delete_table(self):
        item = self.data_display.currentItem()
        if not item:
            return
        if item.childCount():
            table_name = item.text(0)
            reply = QMessageBox.question(self.centralwidget, 'Delete', 'Do you want to detele Table "%s" ?' % table_name
                                         , QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
            if reply == QMessageBox.Yes:
                self.cur.execute('DROP TABLE "%s"' % table_name)
                index = self.data_display.indexOfTopLevelItem(item)
                self.data_display.takeTopLevelItem(index)
            else:
                return
        else:
            return

    def change_table_name(self, item):
        name = item.text(0)
        childwin = ChangeName(name)
        childwin.signal_parp.connect(self.change_name_signal)  # 主窗口接收信号
        childwin.exec_()

    def change_name_signal(self, m):
        self.cur.execute("select name from sqlite_master where type='table'")
        tab_names = [line[0] for line in self.cur.fetchall()]
        item = self.data_display.currentItem()
        name = item.text(0)
        if m in tab_names:
            if m == name:
                return
            else:
                QMessageBox.information(self, "Information", 'Already have Table called "%s"' % m)
                return
        item.setText(0, m)
        self.cur.execute('ALTER TABLE "%s" RENAME TO "%s"' % (name, m))

    def click_to_plot(self):
        if self.plot_lock:
            QMessageBox.information(self, "Information", 'Please wait until result save process finished')
            return
        item = self.data_display.currentItem()
        if item.childCount():
            return
        name = item.text(0)
        table = item.parent().text(0)
        data_info = 'SELECT Raw_Spectrum,Raw_Axis FROM "%s" WHERE Component="%s"' % (table, name)
        all_byte = self.cur.execute(data_info).fetchall()
        data_array = np.array(json.loads(all_byte[0][0]))
        # data_array = data_array / np.max(data_array)
        shift_array = np.array(json.loads(all_byte[0][1]))
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

    def create_models(self, raman_shift=None, model_path=None):
        childwin = TrainNewP(raman_shift, model_path)
        childwin.move(self.geometry().x() + (self.geometry().width() - childwin.width()) // 2,
                      self.geometry().y() + (self.geometry().height() - childwin.height()) // 2)
        childwin.signal_parp.connect(self.get_train_signal)  # 主窗口接收信号
        childwin.exec_()

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
        raman_shift = None
        model_path = None
        fixed = True
        if not item.childCount():
            self.train_table_widget = item.parent()
            self.train_table = item.parent().text(0)
            datas = self.cur.execute('SELECT Component,Raw_Spectrum,Raw_Axis FROM "%s"' % self.train_table).fetchall()
            for data in datas:
                self.train_com_name.append(data[0])
                self.train_com_spec.append(np.array(json.loads(data[1])))
                self.t_axis.append(np.array(json.loads(data[2])))
            if item.text(2) == 'Yes':
                reply = QMessageBox.question(self, 'Train', 'Do you want to re-train "%s"?' % item.text(0),
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if reply == QMessageBox.No:
                    return
                elif reply == QMessageBox.Yes:
                    com = item.text(0)
                    self.cur.execute('UPDATE "%s" SET Get_trained="No",Model_Path="%s",New_Spectrum="%s",New_Axis="%s"'
                                     'WHERE Component="%s"' % (self.train_table, '', '', '', com))
                    self.db.commit()
                    self.train_index = [self.train_com_name.index(item.text(0))]
                else:
                    return
            else:
                self.train_index = [self.train_com_name.index(item.text(0))]
        else:
            self.train_table_widget = item
            self.train_table = item.text(0)
            datas = self.cur.execute('SELECT Component,Raw_Spectrum,Raw_Axis FROM "%s"' % self.train_table).fetchall()
            for data in datas:
                self.train_com_name.append(data[0])
                self.train_com_spec.append(np.array(json.loads(data[1])))
                self.t_axis.append(np.array(json.loads(data[2])))
            jug = self.cur.execute('SELECT Component FROM "%s" WHERE Get_trained="Yes"' % self.train_table).fetchone()
            if jug:
                reply = QMessageBox.question(self, 'Train', "Already have some trained models \n "
                                                            "Do you want to re-train them?",
                                             QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
                if reply == QMessageBox.Yes:
                    self.train_index = list(np.arange(0, len(self.train_com_name)))
                    fixed = False
                elif reply == QMessageBox.No:
                    train_datas = self.cur.execute('SELECT Component FROM "%s" '
                                                   'WHERE Get_trained="No"' % self.train_table).fetchall()
                    if not train_datas:
                        return
                    for data in train_datas:
                        self.train_index.append(self.train_com_name.index(data[0]))
                else:
                    return
            else:
                self.train_index = list(np.arange(0, len(self.train_com_name)))
        old_para = self.cur.execute(
            'SELECT New_Axis,Model_Path FROM "%s" WHERE Get_trained="Yes"' % self.train_table).fetchone()
        if (old_para is not None) & (fixed is True):
            raman_shift = []
            old_axis = np.array(json.loads(old_para[0]))
            raman_shift.append(old_axis[0])
            raman_shift.append(old_axis[-1])
            raman_shift.append(int(old_axis[1] - old_axis[0]))
            model_path = os.path.dirname(old_para[1])
        AppWindow.create_models(self, raman_shift, model_path)

    def get_train_signal(self, m):
        self.model_path = m[-1]
        self.t_new_axis = np.linspace(m[0], m[1], int((m[1] - m[0]) / m[2] + 1))
        AppWindow.train_run(self, m[3:], self.train_index, self.t_axis, self.t_new_axis)

    def train_run(self, sp, count, axis, new_axis):
        optimizer_list = {"Adam": tf.keras.optimizers.Adam(lr=sp[1]),
                          "Adadelta": tf.keras.optimizers.Adadelta(lr=sp[1]),
                          "Adagrad": tf.keras.optimizers.Adagrad(lr=sp[1]),
                          "Adamax": tf.keras.optimizers.Adamax(lr=sp[1])}
        optimizer = optimizer_list[sp[0]]

        train_para = [optimizer, sp[2], sp[3], sp[-1]]
        aug_para = sp[4:6]
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
            childwin = TrainHistory(self.train_para)
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
        table = self.train_table
        spectrum = json.dumps(m[1].tolist())
        new_axis = json.dumps(self.t_new_axis.tolist())
        name = m[0]
        path = os.path.abspath(self.model_path) + '/' + name
        self.cur.execute('UPDATE "%s" SET Get_trained="Yes",Model_Path="%s",New_Spectrum="%s",New_Axis="%s"'
                         'WHERE Component="%s"' % (table, path, spectrum, new_axis, name))
        self.db.commit()
        self.train_table_widget.child(self.train_com_name.index(name)).setText(1, 'Yes')

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
        self.cur.execute("select name from sqlite_master where type='table'")
        tab_name_db = self.cur.fetchall()
        tab_names = [line[0] for line in tab_name_db]
        childwin = PredictionRun(tab_names)
        childwin.signal_parp.connect(self.predict_process)
        childwin.exec_()

    def predict_process(self, m):
        if not m:
            return
        self.pred_table = m
        datas_list = self.cur.execute('SELECT New_Axis,Model_Path FROM "%s" '
                                      'WHERE Get_trained="Yes"' % self.pred_table).fetchall()
        if not datas_list:
            QMessageBox.information(self, "Information", 'No available models')
            return
        model_path = []
        self.axis = np.array(json.loads(datas_list[0][0]))
        for data in datas_list:
            model_path.append(data[1])
        self.pred_data = self.get_pred_data(self.axis)
        if self.pred_data is None:
            QMessageBox.information(self, "Information", 'Missing spectra to be analyzed')
            return
        self.thread_2 = PredRun(self.pred_data, model_path)
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
        components = self.cur.execute('SELECT Component, New_Axis FROM "%s" WHERE Get_trained="Yes"' % self.pred_table)
        component_list = components.fetchall()
        for i in range(len(self.pred_names)):
            num = np.where(self.pred_prob[:, i] > 0.5)[0]
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
        childwin = QuantitativeAnalysis()
        childwin.signal_parp.connect(self.quantitative_analysis)
        childwin.exec_()

    def quantitative_analysis(self, m):
        self.QA_on = True
        components = self.cur.execute('SELECT Component, New_Axis FROM "%s" WHERE Get_trained="Yes"' % self.pred_table)
        component_list = components.fetchall()
        i = 0
        mix = []
        com = []
        for pn in self.pred_names:
            x_size = np.array(json.loads(component_list[0][1])).shape[0]
            Spectrum_data = np.zeros((1, x_size))
            preds = self.result_list[pn]
            for pred in preds:
                spectra_info = self.cur.execute('SELECT New_Spectrum, Inter_Time FROM "%s" WHERE Component="%s"'
                                                % (self.pred_table, pred))
                spectra_info = spectra_info.fetchall()
                spectrum = np.array(json.loads(spectra_info[0][0]))
                try:
                    inter_time = float(spectra_info[0][1])
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
            pred_idx = self.pred_names.index(pred)
            mix_data = self.pred_data[pred_idx]
            # mix_data = mix_data / np.max(mix_data)
            self.fig.axes.plot(self.axis, mix_data, label=pred)
            if component_list:
                for m in range(len(component_list)):
                    spectrum = self.cur.execute('SELECT New_Spectrum FROM "%s"'
                                                'WHERE Component="%s"' % (self.pred_table, component_list[m]))
                    spectrum_data = np.array(json.loads(spectrum.fetchall()[0][0]))
                    # spectrum_data = spectrum_data / np.max(spectrum_data)
                    if self.ratios:
                        spectrum_data = spectrum_data * self.ratios[pred_idx][m]
                    self.fig.axes.plot(self.axis, spectrum_data, label=component_list[m])
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
