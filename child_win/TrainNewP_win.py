# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'parameter_win.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(671, 345)
        self.groupBox = QtWidgets.QGroupBox(Dialog)
        self.groupBox.setGeometry(QtCore.QRect(330, 10, 331, 271))
        self.groupBox.setStyleSheet("")
        self.groupBox.setObjectName("groupBox")
        self.textBrowser_8 = QtWidgets.QTextBrowser(self.groupBox)
        self.textBrowser_8.setGeometry(QtCore.QRect(10, 180, 141, 31))
        self.textBrowser_8.setStyleSheet("background-color: rgb(240, 240, 240);\n"
"border: soild 1px rgb(240, 240, 240);")
        self.textBrowser_8.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_8.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_8.setObjectName("textBrowser_8")
        self.textBrowser_5 = QtWidgets.QTextBrowser(self.groupBox)
        self.textBrowser_5.setGeometry(QtCore.QRect(10, 20, 141, 31))
        self.textBrowser_5.setStyleSheet("background-color: rgb(240, 240, 240);\n"
"border: soild 1px rgb(240, 240, 240);")
        self.textBrowser_5.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_5.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_5.setObjectName("textBrowser_5")
        self.epochs_ = QtWidgets.QSpinBox(self.groupBox)
        self.epochs_.setGeometry(QtCore.QRect(170, 180, 151, 31))
        self.epochs_.setMaximum(999999)
        self.epochs_.setProperty("value", 500)
        self.epochs_.setObjectName("epochs_")
        self.textBrowser_6 = QtWidgets.QTextBrowser(self.groupBox)
        self.textBrowser_6.setGeometry(QtCore.QRect(10, 70, 141, 31))
        self.textBrowser_6.setStyleSheet("background-color: rgb(240, 240, 240);\n"
"border: soild 1px rgb(240, 240, 240);")
        self.textBrowser_6.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_6.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_6.setObjectName("textBrowser_6")
        self.optimizer_ = QtWidgets.QComboBox(self.groupBox)
        self.optimizer_.setGeometry(QtCore.QRect(170, 20, 151, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.optimizer_.setFont(font)
        self.optimizer_.setObjectName("optimizer_")
        self.optimizer_.addItem("")
        self.optimizer_.addItem("")
        self.optimizer_.addItem("")
        self.optimizer_.addItem("")
        self.learnrate = QtWidgets.QLineEdit(self.groupBox)
        self.learnrate.setGeometry(QtCore.QRect(170, 70, 151, 31))
        self.learnrate.setStyleSheet("")
        self.learnrate.setObjectName("learnrate")
        self.textBrowser_7 = QtWidgets.QTextBrowser(self.groupBox)
        self.textBrowser_7.setGeometry(QtCore.QRect(10, 130, 141, 31))
        self.textBrowser_7.setStyleSheet("background-color: rgb(240, 240, 240);\n"
"border: soild 1px rgb(240, 240, 240);")
        self.textBrowser_7.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_7.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_7.setObjectName("textBrowser_7")
        self.batchsize = QtWidgets.QSpinBox(self.groupBox)
        self.batchsize.setGeometry(QtCore.QRect(170, 130, 151, 31))
        self.batchsize.setMaximum(999999)
        self.batchsize.setProperty("value", 512)
        self.batchsize.setObjectName("batchsize")
        self.textBrowser_10 = QtWidgets.QTextBrowser(self.groupBox)
        self.textBrowser_10.setGeometry(QtCore.QRect(10, 230, 141, 31))
        self.textBrowser_10.setStyleSheet("background-color: rgb(240, 240, 240);\n"
"border: soild 1px rgb(240, 240, 240);")
        self.textBrowser_10.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_10.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_10.setObjectName("textBrowser_10")
        self.savepath = QtWidgets.QLineEdit(self.groupBox)
        self.savepath.setGeometry(QtCore.QRect(170, 230, 121, 31))
        self.savepath.setObjectName("savepath")
        self.dir_choose = QtWidgets.QPushButton(self.groupBox)
        self.dir_choose.setGeometry(QtCore.QRect(290, 230, 31, 31))
        self.dir_choose.setObjectName("dir_choose")
        self.groupBox_2 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 120, 311, 161))
        self.groupBox_2.setStyleSheet("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.groupBox_2)
        self.textBrowser_2.setGeometry(QtCore.QRect(10, 70, 131, 31))
        self.textBrowser_2.setStyleSheet("background-color: rgb(240, 240, 240);\n"
"border: soild 1px rgb(240, 240, 240);")
        self.textBrowser_2.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_2.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.interval = QtWidgets.QSpinBox(self.groupBox_2)
        self.interval.setGeometry(QtCore.QRect(150, 120, 151, 31))
        self.interval.setProperty("value", 2)
        self.interval.setObjectName("interval")
        self.textBrowser_3 = QtWidgets.QTextBrowser(self.groupBox_2)
        self.textBrowser_3.setGeometry(QtCore.QRect(10, 120, 131, 31))
        self.textBrowser_3.setStyleSheet("background-color: rgb(240, 240, 240);\n"
"border: soild 1px rgb(240, 240, 240);")
        self.textBrowser_3.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_3.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_3.setObjectName("textBrowser_3")
        self.textBrowser = QtWidgets.QTextBrowser(self.groupBox_2)
        self.textBrowser.setGeometry(QtCore.QRect(10, 20, 131, 31))
        self.textBrowser.setStyleSheet("background-color: rgb(240, 240, 240);\n"
"border: soild 1px rgb(240, 240, 240);")
        self.textBrowser.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser.setObjectName("textBrowser")
        self.startshift = QtWidgets.QSpinBox(self.groupBox_2)
        self.startshift.setGeometry(QtCore.QRect(150, 20, 151, 31))
        self.startshift.setMaximum(99999)
        self.startshift.setProperty("value", 240)
        self.startshift.setObjectName("start shift")
        self.endshift = QtWidgets.QSpinBox(self.groupBox_2)
        self.endshift.setGeometry(QtCore.QRect(150, 70, 151, 31))
        self.endshift.setMaximum(99999)
        self.endshift.setProperty("value", 2000)
        self.endshift.setObjectName("end shift")
        self.save_ = QtWidgets.QPushButton(Dialog)
        self.save_.setEnabled(False)
        self.save_.setGeometry(QtCore.QRect(210, 300, 93, 28))
        self.save_.setObjectName("save_")
        self.reset_ = QtWidgets.QPushButton(Dialog)
        self.reset_.setGeometry(QtCore.QRect(350, 300, 93, 28))
        self.reset_.setObjectName("reset_")
        self.groupBox_4 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_4.setGeometry(QtCore.QRect(10, 10, 311, 111))
        self.groupBox_4.setObjectName("groupBox_4")
        self.textBrowser_9 = QtWidgets.QTextBrowser(self.groupBox_4)
        self.textBrowser_9.setGeometry(QtCore.QRect(10, 70, 131, 31))
        self.textBrowser_9.setStyleSheet("background-color: rgb(240, 240, 240);\n"
"border: soild 1px rgb(240, 240, 240);")
        self.textBrowser_9.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_9.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_9.setObjectName("textBrowser_9")
        self.noise = QtWidgets.QLineEdit(self.groupBox_4)
        self.noise.setGeometry(QtCore.QRect(150, 70, 151, 31))
        self.noise.setStyleSheet("")
        self.noise.setObjectName("noise")
        self.textBrowser_11 = QtWidgets.QTextBrowser(self.groupBox_4)
        self.textBrowser_11.setGeometry(QtCore.QRect(10, 20, 131, 31))
        self.textBrowser_11.setStyleSheet("background-color: rgb(240, 240, 240);\n"
"border: soild 1px rgb(240, 240, 240);")
        self.textBrowser_11.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_11.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_11.setObjectName("textBrowser_11")
        self.number = QtWidgets.QSpinBox(self.groupBox_4)
        self.number.setGeometry(QtCore.QRect(150, 20, 151, 31))
        self.number.setMaximum(999999)
        self.number.setProperty("value", 30000)
        self.number.setObjectName("number")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Parameter"))
        self.groupBox.setTitle(_translate("Dialog", "Training"))
        self.textBrowser_8.setHtml(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">epochs</span></p></body></html>"))
        self.textBrowser_5.setHtml(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">optimizer</span></p></body></html>"))
        self.textBrowser_6.setHtml(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">learning rate</span></p></body></html>"))
        self.optimizer_.setItemText(0, _translate("Dialog", "Adam"))
        self.optimizer_.setItemText(1, _translate("Dialog", "Adadelta"))
        self.optimizer_.setItemText(2, _translate("Dialog", "Adagrad"))
        self.optimizer_.setItemText(3, _translate("Dialog", "Adamax"))
        self.learnrate.setText(_translate("Dialog", "0.00001"))
        self.textBrowser_7.setHtml(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">batch size</span></p></body></html>"))
        self.textBrowser_10.setHtml(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">models path</span></p></body></html>"))
        self.groupBox_2.setTitle(_translate("Dialog", "Raman Shift"))
        self.textBrowser_2.setHtml(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">end</span></p></body></html>"))
        self.textBrowser_3.setHtml(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">interval</span></p></body></html>"))
        self.textBrowser.setHtml(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">start</span></p></body></html>"))
        self.save_.setText(_translate("Dialog", "save"))
        self.reset_.setText(_translate("Dialog", "reset"))
        self.groupBox_4.setTitle(_translate("Dialog", "Data Augmentation"))
        self.textBrowser_9.setHtml(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">noise rate</span></p></body></html>"))
        self.noise.setText(_translate("Dialog", "0.0005"))
        self.textBrowser_11.setHtml(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">number</span></p></body></html>"))
