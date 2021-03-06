from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(288, 244)
        self.groupBox_3 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 0, 271, 141))
        self.groupBox_3.setStyleSheet("")
        self.groupBox_3.setObjectName("groupBox_3")
        self.label_12 = QtWidgets.QLabel(self.groupBox_3)
        self.label_12.setGeometry(QtCore.QRect(10, 20, 101, 31))
        self.label_12.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.groupBox_3)
        self.label_13.setGeometry(QtCore.QRect(10, 60, 101, 31))
        self.label_13.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.groupBox_3)
        self.label_14.setGeometry(QtCore.QRect(10, 100, 101, 31))
        self.label_14.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.label_14.setObjectName("label_14")
        self.startshift = QtWidgets.QDoubleSpinBox(self.groupBox_3)
        self.startshift.setGeometry(QtCore.QRect(110, 20, 161, 31))
        self.startshift.setMaximum(99999.99)
        self.startshift.setSingleStep(0.01)
        self.startshift.setProperty("value", 240.0)
        self.startshift.setObjectName("startshift")
        self.endshift = QtWidgets.QDoubleSpinBox(self.groupBox_3)
        self.endshift.setGeometry(QtCore.QRect(110, 60, 161, 31))
        self.endshift.setMaximum(99999.99)
        self.endshift.setSingleStep(0.01)
        self.endshift.setProperty("value", 2000.0)
        self.endshift.setObjectName("endshift")
        self.interval = QtWidgets.QDoubleSpinBox(self.groupBox_3)
        self.interval.setGeometry(QtCore.QRect(110, 100, 161, 31))
        self.interval.setMaximum(99999.99)
        self.interval.setSingleStep(0.01)
        self.interval.setProperty("value", 2.0)
        self.interval.setObjectName("interval")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(20, 150, 101, 31))
        self.label_2.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.label_2.setObjectName("label_2")
        self.dir_choose = QtWidgets.QPushButton(Dialog)
        self.dir_choose.setGeometry(QtCore.QRect(250, 150, 31, 31))
        self.dir_choose.setText("")
        self.dir_choose.setObjectName("dir_choose")
        self.modelpath = QtWidgets.QLineEdit(Dialog)
        self.modelpath.setGeometry(QtCore.QRect(120, 150, 131, 31))
        self.modelpath.setObjectName("modelpath")
        self.modelpath.setFocusPolicy(QtCore.Qt.NoFocus)
        self.link_ = QtWidgets.QPushButton(Dialog)
        self.link_.setEnabled(True)
        self.link_.setGeometry(QtCore.QRect(30, 200, 93, 28))
        self.link_.setObjectName("link_")
        self.cancel_ = QtWidgets.QPushButton(Dialog)
        self.cancel_.setGeometry(QtCore.QRect(160, 200, 93, 28))
        self.cancel_.setObjectName("cancel_")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Load Models"))
        self.groupBox_3.setTitle(_translate("Dialog", "Raman Shift"))
        self.label_12.setText(_translate("Dialog", "Start"))
        self.label_13.setText(_translate("Dialog", "End"))
        self.label_14.setText(_translate("Dialog", "Interval"))
        self.label_2.setText(_translate("Dialog", "models path"))
        self.link_.setText(_translate("Dialog", "load"))
        self.cancel_.setText(_translate("Dialog", "cancel"))