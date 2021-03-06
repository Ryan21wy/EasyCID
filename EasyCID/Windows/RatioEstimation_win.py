from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_dialog(object):
    def setupUi(self, dialog):
        dialog.setObjectName("dialog")
        dialog.resize(321, 351)
        self.OK = QtWidgets.QPushButton(dialog)
        self.OK.setEnabled(True)
        self.OK.setGeometry(QtCore.QRect(40, 310, 93, 28))
        self.OK.setObjectName("OK")
        self.groupBox_2 = QtWidgets.QGroupBox(dialog)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 0, 301, 191))
        self.groupBox_2.setObjectName("groupBox_2")
        self.checkBox = QtWidgets.QCheckBox(self.groupBox_2)
        self.checkBox.setGeometry(QtCore.QRect(10, 20, 261, 19))
        self.checkBox.setCheckable(True)
        self.checkBox.setObjectName("checkBox")
        self.checkBox_2 = QtWidgets.QCheckBox(self.groupBox_2)
        self.checkBox_2.setGeometry(QtCore.QRect(10, 130, 271, 19))
        self.checkBox_2.setObjectName("checkBox_2")
        self.sb_lamda = QtWidgets.QSpinBox(self.groupBox_2)
        self.sb_lamda.setGeometry(QtCore.QRect(180, 40, 111, 22))
        self.sb_lamda.setFrame(False)
        self.sb_lamda.setMaximum(99999)
        self.sb_lamda.setProperty("value", 10)
        self.sb_lamda.setObjectName("sb_lamda")
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setGeometry(QtCore.QRect(30, 40, 111, 21))
        self.label_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.sb_proder = QtWidgets.QSpinBox(self.groupBox_2)
        self.sb_proder.setGeometry(QtCore.QRect(180, 70, 111, 22))
        self.sb_proder.setFrame(False)
        self.sb_proder.setProperty("value", 1)
        self.sb_proder.setObjectName("sb_proder")
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_3.setGeometry(QtCore.QRect(30, 70, 111, 21))
        self.label_3.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.sb_maxiter = QtWidgets.QSpinBox(self.groupBox_2)
        self.sb_maxiter.setGeometry(QtCore.QRect(180, 100, 111, 22))
        self.sb_maxiter.setFrame(False)
        self.sb_maxiter.setMaximum(99999)
        self.sb_maxiter.setProperty("value", 100)
        self.sb_maxiter.setObjectName("sb_maxiter")
        self.label_4 = QtWidgets.QLabel(self.groupBox_2)
        self.label_4.setGeometry(QtCore.QRect(30, 100, 111, 21))
        self.label_4.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.sm_lamda = QtWidgets.QSpinBox(self.groupBox_2)
        self.sm_lamda.setGeometry(QtCore.QRect(180, 160, 111, 22))
        self.sm_lamda.setFrame(False)
        self.sm_lamda.setMaximum(99999)
        self.sm_lamda.setProperty("value", 1)
        self.sm_lamda.setObjectName("sm_lamda")
        self.label_5 = QtWidgets.QLabel(self.groupBox_2)
        self.label_5.setGeometry(QtCore.QRect(30, 160, 111, 21))
        self.label_5.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.groupBox = QtWidgets.QGroupBox(dialog)
        self.groupBox.setGeometry(QtCore.QRect(10, 190, 301, 111))
        self.groupBox.setFlat(False)
        self.groupBox.setObjectName("groupBox")
        self.EN_iter = QtWidgets.QSpinBox(self.groupBox)
        self.EN_iter.setGeometry(QtCore.QRect(180, 80, 111, 22))
        self.EN_iter.setFrame(False)
        self.EN_iter.setMaximum(99999)
        self.EN_iter.setProperty("value", 100)
        self.EN_iter.setObjectName("EN_iter")
        self.label_6 = QtWidgets.QLabel(self.groupBox)
        self.label_6.setGeometry(QtCore.QRect(30, 80, 111, 21))
        self.label_6.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setGeometry(QtCore.QRect(30, 50, 111, 21))
        self.label_7.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.groupBox)
        self.label_8.setGeometry(QtCore.QRect(10, 20, 201, 31))
        self.label_8.setStyleSheet("")
        self.label_8.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_8.setTextFormat(QtCore.Qt.AutoText)
        self.label_8.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_8.setWordWrap(False)
        self.label_8.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.label_8.setObjectName("label_8")
        self.EN_ratio = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.EN_ratio.setGeometry(QtCore.QRect(180, 50, 111, 22))
        self.EN_ratio.setFrame(False)
        self.EN_ratio.setPrefix("")
        self.EN_ratio.setMaximum(1.0)
        self.EN_ratio.setSingleStep(0.01)
        self.EN_ratio.setProperty("value", 0.96)
        self.EN_ratio.setObjectName("EN_ratio")
        self.cancel_ = QtWidgets.QPushButton(dialog)
        self.cancel_.setEnabled(True)
        self.cancel_.setGeometry(QtCore.QRect(180, 310, 93, 28))
        self.cancel_.setObjectName("cancel_")

        self.retranslateUi(dialog)
        QtCore.QMetaObject.connectSlotsByName(dialog)

    def retranslateUi(self, dialog):
        _translate = QtCore.QCoreApplication.translate
        dialog.setWindowTitle(_translate("dialog", "Ratio Estimaiton Setting"))
        self.OK.setText(_translate("dialog", "OK"))
        self.groupBox_2.setTitle(_translate("dialog", "Pre-Processing"))
        self.checkBox.setText(_translate("dialog", "Baseline Subtracted -- airPLS"))
        self.checkBox_2.setText(_translate("dialog", "Spectral Smoothing -- Whittaker"))
        self.label_2.setText(_translate("dialog", "lamda"))
        self.label_3.setText(_translate("dialog", "proder"))
        self.label_4.setText(_translate("dialog", "max iter"))
        self.label_5.setText(_translate("dialog", "lamda"))
        self.groupBox.setTitle(_translate("dialog", "Ratio Estimation"))
        self.label_6.setText(_translate("dialog", "iter"))
        self.label_7.setText(_translate("dialog", "L1 ratio"))
        self.label_8.setText(_translate("dialog", "Non-negative Elastic Net"))
        self.cancel_.setText(_translate("dialog", "cancel"))
