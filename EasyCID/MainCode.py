import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

import webbrowser
import sqlite3
import numpy as np
import json
import matplotlib

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
from datetime import datetime
# local packages
from EasyCID.readFile.spc.spcio import readSPC
from EasyCID.readFile import jcamp
from EasyCID.readFile.readTXT import parseBWTekFile, read_simple_txt
from EasyCID.Training.cnn_model_tf import cnn_model
from EasyCID.Training.data_augmentation import data_augment
from EasyCID.Training.spilt_dataset import spilt_dataset
from EasyCID.Prediction.AirPLS import airPLS, WhittakerSmooth
from EasyCID.Utils.makeDir import mkdir
from EasyCID.Utils.makeEXCEL import make_excel
from EasyCID.Utils.database import EasyCIDDatabase as EasyDB
# windows of EasyCID
from EasyCID.Windows.MainWindow import Ui_MainWindow
from EasyCID.Windows import TableName_win, PredictionParameter_win, LinkModels_win, TrainingReport_win, \
    RatioEstimation_win, TrainingParameter_win


path_config = {'Setup': '', 'OpenDB': '', 'OpenFiles': '', 'Import': '',
               'SaveResult': '', 'Models': '', 'Augment': '', 'LinkModels': ''}


class TrainingParameterSetting(QDialog):
    signal_parp = pyqtSignal(list)

    def __init__(self, GroupMI=None, ComponentMI=None, fixed=False):
        QDialog.__init__(self)
        self.child = TrainingParameter_win.Ui_Dialog()
        self.child.setupUi(self)
        self.child.dir_choose.setIcon(QIcon('Icon/view.png'))
        self.child.aug_choose.setIcon(QIcon('Icon/view.png'))
        self.child.dir_choose.clicked.connect(self.model_path_chose)
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

    def model_path_chose(self):
        global path_config
        last_path = path_config['Models']
        if last_path:
            save_path = QFileDialog.getExistingDirectory(win.centralwidget, "models save path", last_path)
        else:
            save_path = QFileDialog.getExistingDirectory(win.centralwidget, "models save path", "C:/")
        if not save_path:
            return
        path_config['Models'] = os.path.dirname(save_path)
        self.child.savepath.setText(save_path)

    def aug_chose(self):
        global path_config
        last_path = path_config['Augment']
        if last_path:
            save_path = QFileDialog.getExistingDirectory(win.centralwidget, "augmentation path", last_path)
        else:
            save_path = QFileDialog.getExistingDirectory(win.centralwidget, "augmentation path", "C:/")
        if not save_path:
            return
        path_config['Augment'] = os.path.dirname(save_path)
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
        TrainingParameterSetting.close(self)

    def cancel(self):
        TrainingParameterSetting.close(self)


class TrainingReport(QDialog):
    def __init__(self, parameters):
        QDialog.__init__(self)
        self.child = TrainingReport_win.Ui_Dialog()
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
        TrainingReport.load(self)

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
        self.child.dir_choose.setIcon(QIcon('Icon/view.png'))
        self.child.dir_choose.clicked.connect(self.dir_chose)
        self.group = old_para[-1]
        self.child.link_.clicked.connect(self.link)
        self.child.cancel_.clicked.connect(self.cancel)

    def dir_chose(self):
        global path_config
        last_path = path_config['LinkModels']
        if last_path:
            model_path = QFileDialog.getExistingDirectory(win.centralwidget, "choose model path", last_path)
        else:
            model_path = QFileDialog.getExistingDirectory(win.centralwidget, "choose model path", "C:/")
        if not model_path:
            return
        path_config['LinkModels'] = os.path.dirname(model_path)
        self.child.modelpath.setText(model_path)
        info_path = os.path.join(model_path, 'ModelsInfo.json')
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r') as fp:
                    info = json.load(fp)
                self.child.startshift.setValue(info['start'])
                self.child.endshift.setValue(info['end'])
                self.child.interval.setValue(info['interval'])
            except:
                return
        else:
            return

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


class RatioEstimationSetting(QDialog):
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
        RatioEstimationSetting.close(self)

    def cancel(self):
        RatioEstimationSetting.close(self)


class PredictionSetting(QDialog):
    signal_parp = pyqtSignal(list)

    def __init__(self, table_list):
        QDialog.__init__(self)
        self.child = PredictionParameter_win.Ui_dialog()
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


class TrainingRun(QThread):
    process_signal = pyqtSignal(str)
    max_bar = pyqtSignal(int)
    current_bar = pyqtSignal(int)
    bar_text = pyqtSignal(str)
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

    def run(self):
        try:
            self.signal.emit('run')
            Spectrumdata = np.zeros((1, self.new_axis.shape[0]))
            count = len(self.spectra)
            new_spectra = []
            for i in range(count):
                spectrum = np.interp(self.new_axis, self.axis[i], self.spectra[i]).astype(np.float64).copy()
                new_spectra.append([self.names[i], spectrum])
                spectrum = spectrum / np.max(spectrum)
                Spectrumdata = np.vstack((Spectrumdata, spectrum.T))
            spectra_raw = np.delete(Spectrumdata, 0, 0)
            current_com = 0
            total_com = len(self.count)
            self.process_signal.emit('Training Active')
            self.current_bar.emit(current_com)
            self.bar_text.emit('%s/%s' % (current_com, total_com))
            self.max_bar.emit(total_com)
            t0 = datetime.now()
            for com in self.count:
                para_record = []
                if self.aug_save_path:
                    mkdir(self.aug_save_path)
                    aug_data_path = os.path.join(self.aug_save_path, self.names[com] + '.npy')
                    aug_label_path = os.path.join(self.aug_save_path, self.names[com] + '_label.npy')
                    if os.path.isfile(aug_data_path):
                        spectrum = np.load(aug_data_path)
                        label = np.load(aug_label_path)
                    else:
                        spectrum, label = data_augment(spectra_raw, com, num=self.aug_number, nr=self.noise_rate)
                        np.save(aug_data_path, spectrum)
                        np.save(aug_label_path, label)
                else:
                    spectrum, label = data_augment(spectra_raw, com, num=self.aug_number, nr=self.noise_rate)
                Xtrain, Xtest, Ytrain, Ytest = spilt_dataset(spectrum, label)
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
                para_record.append(self.names[com])
                para_record.append(history.history['loss'])
                para_record.append(history.history['accuracy'])
                para_record.append(history.history['val_loss'])
                para_record.append(history.history['val_accuracy'])
                para_record.append(loss)
                para_record.append(acc)
                mkdir(self.model_path)
                model.save_weights(os.path.join(self.model_path, self.names[com] + '.h5'))

                del model
                self.data_signal.emit(new_spectra[com])
                self.para_signal.emit(para_record)
                current_com += 1
                t1 = datetime.now()
                cost_t = str(t1 - t0).split('.')[0]
                average_t = str(((t1 - t0) / current_com).seconds)
                remain_t = str((t1 - t0) / current_com * (total_com - current_com)).split('.')[0]
                self.current_bar.emit(current_com)
                self.bar_text.emit('%s/%s' % (current_com, total_com))
                self.process_signal.emit('Training Active [%s<%s] %ss/it' % (cost_t, remain_t, average_t))
            self.signal.emit('finished')
        except Exception as err:
            self.err_signal.emit(str(err))


class PredictionRun(QThread):
    max_bar = pyqtSignal(int)
    current_bar = pyqtSignal(int)
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
            current_model = 0
            total_model = len(self.model_path)
            self.current_bar.emit(current_model)
            self.max_bar.emit(total_model)
            for path in self.model_path:
                tf.keras.backend.clear_session()
                ops.reset_default_graph()
                reload_model.load_weights(path)
                Xtest_pre = Xtest_pre.reshape(Xtest_pre.shape[0], Xtest_pre.shape[1], 1)
                y = reload_model.predict(Xtest_pre)
                y_DeepCID.append(y)
                current_model += 1
                self.current_bar.emit(current_model)
            self.data_signal.emit(y_DeepCID)
            self.signal.emit('finished')
        except Exception as err:
            self.signal.emit(str(err))


class RatioEstimationRun(QThread):
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
            make_excel(self.pred_names, self.pred_list, self.save_path, ratios=self.c_list)
            self.signal.emit('finish')
        except Exception as err:
            self.signal.emit(str(err))


class Myplot(FigureCanvas):
    def __init__(self, dpi=100):
        plt.rc('font', family='Times New Roman')
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
        clear_ = QAction(QIcon("Icon/clear.png"), 'Clear plot area', self)
        clear_.triggered.connect(self.erase_plot_func)
        action_.addAction(clear_)
        collect_ = QAction(QIcon('Icon/collect.png'), 'Multiple plot', self)
        collect_.triggered.connect(self.mutiplot_func)
        action_.addAction(collect_)
        func_ = self.addToolBar('Function')
        add_ = QAction(QIcon('Icon/add.png'), 'Train CNN models', self)
        add_.triggered.connect(self.train_models)
        func_.addAction(add_)
        run_ = QAction(QIcon('Icon/predict.png'), 'Prediction', self)
        run_.triggered.connect(self.predict_process_func)
        func_.addAction(run_)
        self.qa = QAction(QIcon('Icon/analysis.png'), 'Quantitative analysis', self)
        self.qa.triggered.connect(self.ratio_estimation_func)
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
        self.action_link.triggered.connect(self.open_db)
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
        self.candidate_model = []
        self.raman_shift_para = []
        self.train_com_name = []
        self.train_com_spec = []
        self.train_para = {}
        self.ratios = None
        self.muti = False
        self.pred_on = False
        self.train_on = False
        self.RE_on = False
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

        if tf.config.experimental.list_physical_devices('GPU'):
            gpus = tf.config.list_physical_devices(device_type='GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(device=gpu, enable=True)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.get_config()

    def get_config(self):
        global path_config
        info_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PathConfig.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as fp:
                path_config = json.load(fp)

    def help_html(self):
        path = 'file:///' + os.path.abspath('helpHTML/index.html')
        webbrowser.open_new_tab(path)

    def connect_db(self, path):
        try:
            self.EasyDB = EasyDB(path)
            self.db = self.EasyDB.db
            self.cur = self.EasyDB.cur
            abs_bath = os.path.abspath(path)
            self.modelPath.setTitle('Current Database: "%s"' % abs_bath)
            AppWindow.db_data_display(self)
        except Exception as err:
            QMessageBox.information(self, "Error", str(err))

    def set_up_db(self):
        global path_config
        last_path = path_config['Setup']
        if last_path:
            file_name, _ = QFileDialog.getSaveFileName(self.centralwidget, "Build database", last_path,
                                                       'database (*.db)')
        else:
            file_name, _ = QFileDialog.getSaveFileName(self.centralwidget, "Build database", "C:/",
                                                       'database (*.db)')
        if not file_name:
            return
        path_config['Setup'] = os.path.dirname(file_name)
        EasyDB(file_name).set_up_database()
        AppWindow.connect_db(self, file_name)

    def open_db(self):
        global path_config
        last_path = path_config['OpenDB']
        if last_path:
            file_name, type = QFileDialog.getOpenFileName(self.centralwidget, "Choose database",
                                                          last_path, 'database (*.db)')
        else:
            file_name, type = QFileDialog.getOpenFileName(self.centralwidget, "Choose database",
                                                          r"C:/", 'database (*.db)')
        if not file_name:
            return
        path_config['OpenDB'] = os.path.dirname(file_name)
        AppWindow.connect_db(self, file_name)

    def get_db_data(self):
        group_db = self.EasyDB.select('*', 'Groups')
        group_names = [line[1] for line in group_db]
        group_ids = [line[0] for line in group_db]
        if not group_names:
            return None, None
        components_info = []
        for id in group_ids:
            component_info = self.EasyDB.select('Component_Name,Model', 'Component_Info', 'From_Group=?', (id,))
            components_info.append(component_info)
        return group_names, components_info

    def db_data_display(self):
        if not self.db:
            return
        group_names, components_info = AppWindow.get_db_data(self)
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

    def open_dir_func(self):
        global path_config
        last_path = path_config['OpenFiles']
        if last_path:
            self.data_path = QFileDialog.getExistingDirectory(self.centralwidget, "选取文件夹", last_path)
        else:
            self.data_path = QFileDialog.getExistingDirectory(self.centralwidget, "选取文件夹", "C:/")
        if not self.data_path:
            return
        path_config['OpenFiles'] = os.path.dirname(self.data_path)
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
                        readfile = parseBWTekFile(newfile, s, select_ramanshift=False,
                                                  xname='Raman Shift', yname="Dark Subtracted #1")
                        raw_axis = readfile['axis']
                        raw_spectrum = readfile['spectrum']
                        raw_axis, raw_spectrum = zip(*sorted(zip(raw_axis, raw_spectrum)))
                        name = readfile['name']
                        inter_time = readfile['integral_time']
                        datas.append([None, name, json.dumps(list(raw_spectrum)),
                                      json.dumps(list(raw_axis)), inter_time, 0])
                    except:
                        raw_axis, raw_spectrum = read_simple_txt(newfile)
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
        global path_config
        if not self.db:
            return
        last_path = path_config['Import']
        if last_path:
            spectra_path = QFileDialog.getExistingDirectory(self.centralwidget, "Choose data path", last_path)
        else:
            spectra_path = QFileDialog.getExistingDirectory(self.centralwidget, "Choose data path", "C:/")
        if not spectra_path:
            return
        path_config['Import'] = os.path.dirname(spectra_path)
        datas = AppWindow.read_spectra(self, spectra_path)
        if not datas:
            QMessageBox.information(self, "Information", 'No available spectral data were found')
            return
        group_db = self.EasyDB.select('Group_Name', 'Groups')
        name_list = [line[0] for line in group_db]
        num = 0
        group_name = 'Models' + str(num)
        while group_name in name_list:
            num += 1
            group_name = 'Models' + str(num)
        self.EasyDB.insert('Groups', '(?,?)', (None, group_name))
        group_id = self.EasyDB.select('Group_ID', 'Groups', 'Group_Name=?', (group_name,))[0][0]
        for i in range(len(datas)):
            datas[i].append(group_id)
        self.EasyDB.insert('Component_Info', '(?,?,?,?,?,?,?)', datas, many=True)
        root = QTreeWidgetItem(self.data_display)
        root.setText(0, group_name)
        for column in datas:
            child = QTreeWidgetItem()
            child.setText(0, column[1])
            child.setText(1, self.model_ref[column[5]])
            root.addChild(child)

    def add_spectra(self, item):
        global path_config
        if not item.childCount():
            return
        last_path = path_config['Import']
        if last_path:
            spectra_path = QFileDialog.getExistingDirectory(self.centralwidget, "选取文件夹", last_path)
        else:
            spectra_path = QFileDialog.getExistingDirectory(self.centralwidget, "选取文件夹", "C:/")
        if not spectra_path:
            return
        path_config['Import'] = os.path.dirname(spectra_path)
        datas = AppWindow.read_spectra(self, spectra_path)
        if not datas:
            QMessageBox.information(self, "Information", 'No available spectral data were found')
            return
        group_id = self.EasyDB.select('Group_ID', 'Groups', 'Group_Name=?', (item.text(0),))[0][0]
        for i in range(len(datas)):
            datas[i].append(group_id)
        self.EasyDB.insert('Component_Info', '(?,?,?,?,?,?,?)', datas, many=True)
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
                current_id = self.EasyDB.select('Group_ID', 'Groups', 'Group_Name=?', (group_name,))[0][0]
                self.EasyDB.delete('Component_Info', 'Component_Name=? and From_Group=?', (name, current_id))
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
                self.EasyDB.delete('Groups', 'Group_Name=?', (group_name,))
                index = self.data_display.indexOfTopLevelItem(item)
                self.data_display.takeTopLevelItem(index)
            else:
                return
        else:
            return

    def change_group_name(self, item):
        name = item.text(0)
        childwin = ChangeName(name)
        childwin.signal_parp.connect(self.change_name_signal)
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
        current_id = self.EasyDB.select('Group_ID', 'Groups', 'Group_Name=?', (name,))[0][0]
        self.EasyDB.update('Groups', 'Group_Name=?', 'Group_ID=?', (m, current_id))

    def link_Models(self, item):
        self.link_widget = item
        group = item.text(0)
        group_id = self.EasyDB.select('Group_ID', 'Groups', 'Group_Name=?', (group,))[0][0]
        old_para = self.EasyDB.select('*', 'Group_Model_Info', 'From_Group=?', (group_id,))
        if not old_para:
            old_para = [group_id]
        else:
            old_para = old_para[0]
        childwin = LinkModels(old_para=old_para)
        childwin.move(self.geometry().x() + (self.geometry().width() - childwin.width()) // 2,
                      self.geometry().y() + (self.geometry().height() - childwin.height()) // 2)
        childwin.signal_parp.connect(self.get_link_signal)
        childwin.exec_()

    def get_link_signal(self, m):
        QApplication.processEvents()
        group_id = m[0]
        correct_models = m[-1]
        names_db = self.EasyDB.select('Component_Name', 'Component_Info', 'From_Group=?', (group_id,))
        names = [n[0] for n in names_db]
        for name in names:
            if name in correct_models:
                self.EasyDB.update('Component_Info', 'Model=?', 'Component_Name=? and From_Group=?',
                                   (1, name, group_id))
                self.link_widget.child(names.index(name)).setText(1, self.model_ref[1])
            else:
                self.EasyDB.update('Component_Info', 'Model=?', 'Component_Name=? and From_Group=?',
                                   (0, name, group_id))
                self.link_widget.child(names.index(name)).setText(1, self.model_ref[0])
        old_para = self.EasyDB.select('*', 'Group_Model_Info', 'From_Group=?', (group_id,))
        if not old_para:
            self.EasyDB.insert('Group_Model_Info', '(?,?,?,?,?,?)', (m[1], m[2], m[3], '', m[4], group_id))
        else:
            self.EasyDB.update('Group_Model_Info', 'Raman_Start=?, Raman_End=?, Raman_Interval=?, Save_Path=?',
                               'From_Group=?', (m[1], m[2], m[3], m[4], group_id))
        QMessageBox.information(self, "Information", 'Complete Load Models')

    def click_to_plot(self):
        if self.plot_lock:
            QMessageBox.information(self, "Information", 'Please wait until result save process finished')
            return
        item = self.data_display.currentItem()
        if item.childCount():
            return
        name = item.text(0)
        group = item.parent().text(0)
        group_id = self.EasyDB.select('Group_ID', 'Groups', 'Group_Name=?', (group,))[0][0]
        datas = self.EasyDB.select('Raw_Spectrum,Raw_Axis', 'Component_Info', 'Component_Name=? and From_Group=?',
                                   (name, group_id))[0]
        data_array = np.array(json.loads(datas[0]))
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
            group_id = self.EasyDB.select('Group_ID', 'Groups', 'Group_Name=?', (self.train_group,))[0][0]
            names = self.EasyDB.select('Component_Name', 'Component_Info', 'From_Group=?', (group_id,))
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
                index = self.EasyDB.select('Component_ID', 'Component_Info', 'Component_Name=? and From_Group=?',
                                           (item.text(0), group_id))[0][0]
                try:
                    ComponentMI = self.EasyDB.select('*', 'Component_Model_Info', 'From_Component=?', (index,))[0]
                except:
                    ComponentMI = None
                jug = self.EasyDB.select('Component_ID', 'Component_Info', 'Model=1 and From_Group=?', (group_id,))
                if len(jug) >= 2:
                    fixed = True
        else:
            self.train_group_widget = item
            self.train_group = item.text(0)
            group_id = self.EasyDB.select('Group_ID', 'Groups', 'Group_Name=?', (self.train_group,))[0][0]
            names = self.EasyDB.select('Component_Name', 'Component_Info', 'From_Group=?', (group_id,))
            for name in names:
                self.train_com_name.append(name[0])
            jug = self.EasyDB.select('Component_Name', 'Component_Info', 'Model=1 and From_Group=?', (group_id,))
            if jug:
                reply = QMessageBox.question(self, 'Train', "Already have some trained models \n "
                                                            "Do you want to re-train them?",
                                             QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
                if reply == QMessageBox.Yes:
                    self.train_index = list(np.arange(0, len(self.train_com_name)))
                elif reply == QMessageBox.No:
                    components = self.EasyDB.select('Component_Name', 'Component_Info', 'Model=0 and From_Group=?',
                                                    (group_id,))
                    if not components:
                        return
                    for com in components:
                        self.train_index.append(self.train_com_name.index(com[0]))
                    fixed = True
                else:
                    return
            else:
                self.train_index = list(np.arange(0, len(self.train_com_name)))
        datas = self.EasyDB.select('Raw_Spectrum, Raw_Axis', 'Component_Info', 'From_Group=?', (group_id,))
        for data in datas:
            self.train_com_spec.append(np.array(json.loads(data[0])))
            self.t_axis.append(np.array(json.loads(data[1])))
        old_para = self.EasyDB.select('*', 'Group_Model_Info', 'From_Group=?', (group_id,))
        if old_para:
            GroupMI = old_para[0]
        self.create_models(GroupMI=GroupMI, ComponentMI=ComponentMI, fixed=fixed)

    def create_models(self, GroupMI=None, ComponentMI=None, fixed=False):
        childwin = TrainingParameterSetting(GroupMI=GroupMI, ComponentMI=ComponentMI, fixed=fixed)
        childwin.move(self.geometry().x() + (self.geometry().width() - childwin.width()) // 2,
                      self.geometry().y() + (self.geometry().height() - childwin.height()) // 2)
        childwin.signal_parp.connect(self.get_train_signal)
        childwin.exec_()

    def get_train_signal(self, m):
        self.GroupMI = [m[0], m[1], m[2], m[9], m[10]]
        self.ComponentMI = [m[7], m[8], m[3], m[4], m[5], m[6]]
        self.model_path = m[-1]
        self.t_new_axis = np.linspace(m[0], m[1], int((m[1] - m[0]) / m[2] + 1))
        model_info = {'start': m[0], 'end': m[1], 'interval': m[2]}
        info_path = os.path.join(self.model_path, 'ModelsInfo.json')
        with open(info_path, 'w') as fp:
            json.dump(model_info, fp)
        AppWindow.train_run(self, m[3:], self.train_index, self.t_axis, self.t_new_axis)

    def train_run(self, sp, count, axis, new_axis):
        optimizer_list = [tf.keras.optimizers.Adam(lr=sp[1]), tf.keras.optimizers.Adadelta(lr=sp[1]),
                          tf.keras.optimizers.Adagrad(lr=sp[1]), tf.keras.optimizers.Adamax(lr=sp[1])]
        optimizer = optimizer_list[sp[0]]
        train_para = [optimizer, sp[2], sp[3], sp[-1]]
        aug_para = sp[4:7]
        info_para = [self.train_com_name, count, self.train_com_spec, axis, new_axis]

        self.thread = TrainingRun(train_para, aug_para, info_para)
        self.thread.signal.connect(self.get_train_thread_signal)
        self.thread.process_signal.connect(self.get_train_process_signal)
        self.thread.max_bar.connect(self.get_max_bar_value)
        self.thread.current_bar.connect(self.get_current_bar_value)
        self.thread.bar_text.connect(self.get_bar_text)
        self.thread.err_signal.connect(self.get_train_err_signal)
        self.thread.data_signal.connect(self.get_train_data_signal)
        self.thread.para_signal.connect(self.get_train_para_signal)
        self.thread.daemon = True
        self.thread.start()

    def get_train_thread_signal(self, m):
        if m == 'run':
            # self.progressBar.setMinimum(0)
            # self.progressBar.setMaximum(0)
            # self.textBrowser_3.setText('Training CNN models')
            self.progressBar.setTextVisible(True)
            self.train_on = True
            self.train_para = {}
        elif m == 'finished':
            self.progressBar.setValue(0)
            self.progressBar.setTextVisible(False)
            self.textBrowser_3.setText('Finished')
            self.train_on = False
            QtCore.QTimer().singleShot(2000, self.clear_text_1)
            childwin = TrainingReport(self.train_para)
            childwin.exec_()
        else:
            QMessageBox.information(self, "Information", m)

    def get_max_bar_value(self, m):
        self.progressBar.setMaximum(m)

    def get_current_bar_value(self, m):
        self.progressBar.setValue(m)

    def get_bar_text(self, m):
        self.progressBar.setFormat(m + ' %p%')

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
        group_id = self.EasyDB.select('Group_ID', 'Groups', 'Group_Name=?', (group,))[0][0]
        self.EasyDB.update('Component_Info', 'Model=?', 'Component_Name=? and From_Group=?', (1, name, group_id))
        GroupMI = self.GroupMI.copy()
        GroupMI.append(group_id)
        jug = self.EasyDB.select('*', 'Group_Model_Info', 'From_Group=?', (group_id,))
        if jug:
            self.EasyDB.update('Group_Model_Info',
                               'Raman_Start=?, Raman_End=?, Raman_Interval=?, Aug_Save_Path=?, Save_Path=?',
                               'From_Group=?', tuple(GroupMI))
        else:
            self.EasyDB.insert('Group_Model_Info', '(?,?,?,?,?,?)', GroupMI)
        component_id = self.EasyDB.select('Component_ID ', 'Component_Info', 'Component_Name=?', (name,))[0][0]
        ComponentMI = self.ComponentMI.copy()
        ComponentMI.append(component_id)
        jug = self.EasyDB.select('*', 'Component_Model_Info', 'From_Component=?', (component_id,))
        if jug:
            self.EasyDB.update('Component_Model_Info', 'Augment_Num=?, Noise_Rate=?, Optimizer=?, LR=?, BS=?, EPS=?',
                               'From_Component=?', tuple(ComponentMI))
        else:
            self.EasyDB.insert('Component_Model_Info', '(?,?,?,?,?,?,?)', ComponentMI)
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
        tab_name_db = self.EasyDB.select('Group_Name', 'Groups')
        tab_names = [line[0] for line in tab_name_db]
        childwin = PredictionSetting(tab_names)
        childwin.signal_parp.connect(self.predict_process)
        childwin.exec_()

    def predict_process(self, m):
        if not m:
            return
        self.pred_group = m[0]
        self.threshold = m[1]
        group_id = self.EasyDB.select('Group_ID', 'Groups', 'Group_Name=?', (self.pred_group,))[0][0]
        component_db = self.EasyDB.select('Component_Name', 'Component_Info', 'Model=1 and From_Group=?', (group_id,))
        component_list = [c[0] for c in component_db]
        group_para = self.EasyDB.select('*', 'Group_Model_Info', 'From_Group=?', (group_id,))[0]
        if not group_para:
            QMessageBox.information(self, "Information", 'No available models')
            return
        self.axis = np.arange(group_para[0], group_para[1], group_para[2])
        self.candidate_model = []
        model_path_list = []
        dir = os.listdir(group_para[-2])
        for file in dir:
            if os.path.isfile(os.path.join(group_para[-2], file)):
                [name, type] = file.split('.')
                if (name in component_list) and (type == 'h5'):
                    model_path_list.append(os.path.join(group_para[-2], file))
                    self.candidate_model.append(name)
        if not model_path_list:
            QMessageBox.information(self, "Information", 'No available models')
            return
        self.pred_data = self.get_pred_data(self.axis)
        if self.pred_data is None:
            QMessageBox.information(self, "Information", 'Missing spectra to be analyzed')
            return
        self.thread_2 = PredictionRun(self.pred_data, model_path_list)
        self.thread_2.signal.connect(self.get_pred_thread_signal)
        self.thread_2.current_bar.connect(self.get_current_bar_value2)
        self.thread_2.max_bar.connect(self.get_max_bar_value2)
        self.thread_2.data_signal.connect(self.get_pred_data_signal)
        self.thread_2.daemon = True
        self.thread_2.start()

    def get_max_bar_value2(self, m):
        self.progressBar_2.setMaximum(m)

    def get_current_bar_value2(self, m):
        self.progressBar_2.setValue(m)

    def get_pred_thread_signal(self, m):
        if m == 'run':
            if self.RE_on:
                self.textBrowser_4.setText('Quantitative analysis')
            else:
                self.progressBar_2.setTextVisible(True)
                self.textBrowser_4.setText('Prediction')
            self.pred_on = True
        elif m == 'finished':
            self.progressBar_2.setValue(0)
            self.progressBar_2.setTextVisible(False)
            self.textBrowser_4.setText('Finished')
            self.pred_on = False
            self.qa.setEnabled(True)
            self.ratios = None
            QtCore.QTimer().singleShot(2000, self.clear_text_2)
        else:
            QMessageBox.information(self, "Information", m)
            self.progressBar_2.setValue(0)
            self.progressBar_2.setTextVisible(False)
            self.textBrowser_4.setText('')
            self.pred_on = False
            self.RE_on = False

    def get_pred_data_signal(self, m):
        self.pred_prob = np.asarray(m)
        self.result_list = {}
        for i in range(len(self.pred_names)):
            num = np.where(self.pred_prob[:, i] >= self.threshold)[0]
            self.result_list[self.pred_names[i]] = [self.candidate_model[n] for n in num]
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

    def ratio_estimation_func(self):
        if self.RE_on:
            QMessageBox.information(self, "Information", 'A prediction process is running')
            return
        if not self.db:
            return
        childwin = RatioEstimationSetting()
        childwin.signal_parp.connect(self.ratios_estimation)
        childwin.exec_()

    def ratios_estimation(self, m):
        self.RE_on = True
        group_id = self.EasyDB.select('Group_ID', 'Groups', 'Group_Name=?', (self.pred_group,))[0][0]
        i = 0
        mix = []
        com = []
        for pn in self.pred_names:
            x_size = self.axis.shape[0]
            Spectrum_data = np.zeros((1, x_size))
            preds = self.result_list[pn]
            for pred in preds:
                spectra_info = self.EasyDB.select('Raw_Axis, Raw_Spectrum, Inter_time', 'Component_Info',
                                                  'Component_Name=? and From_Group=?', (pred, group_id))[0]
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
        self.thread_3 = RatioEstimationRun(mix, com, m)
        self.thread_3.signal.connect(self.get_pred_thread_signal)
        self.thread_3.rate_signal.connect(self.ratios_display)
        self.thread_3.daemon = True
        self.thread_3.start()

    def ratios_display(self, ratios):
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
        self.RE_on = False

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
            group_id = self.EasyDB.select('Group_ID', 'Groups', 'Group_Name=?', (self.pred_group,))[0][0]
            pred_idx = self.pred_names.index(pred)
            mix_data = self.pred_data[pred_idx]
            self.fig.axes.plot(self.axis, mix_data, label=pred)
            if component_list:
                for m in range(len(component_list)):
                    spectra_info = self.EasyDB.select('Raw_Axis, Raw_Spectrum, Inter_time', 'Component_Info',
                                                      'Component_Name=? and From_Group=?',
                                                      (component_list[m], group_id))[0]
                    old_axis = np.array(json.loads(spectra_info[0]))
                    old_spectrum = np.array(json.loads(spectra_info[1]))
                    spectrum = np.interp(self.axis, old_axis, old_spectrum).astype(np.float64)
                    if self.ratios:
                        spectrum = spectrum * self.ratios[pred_idx][m]
                    self.fig.axes.plot(self.axis, spectrum, label=component_list[m])
                self.fig.axes.legend(loc='best')
        self.fig.draw()

    def save_function(self):
        global path_config
        if not self.result_list:
            return
        last_path = path_config['SaveResult']
        if last_path:
            save_path, ext = QFileDialog.getSaveFileName(self.centralwidget, "Choose result save path", last_path,
                                                         "EXCEL(*.xlsx)")
        else:
            save_path, ext = QFileDialog.getSaveFileName(self.centralwidget, "Choose result save path", "C:/",
                                                         "EXCEL(*.xlsx)")
        if not save_path:
            return
        path_config['SaveResult'] = os.path.dirname(save_path)
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
            global path_config
            info_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PathConfig.json')
            with open(info_path, 'w') as fp:
                json.dump(path_config, fp)
            event.accept()
        elif messageBox.clickedButton() == Qno:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = AppWindow()
    win.show()
    sys.exit(app.exec_())
