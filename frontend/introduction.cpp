#include "introduction.h"
#include <QLabel>

Introduction::Introduction(QWidget *parent) : QMainWindow(parent)
{
    //size
    this->setFixedSize(700,400);
    //logo
    this->setWindowIcon(QIcon(":/scene/logo.jpg"));
    QString qstr = QString::fromLocal8Bit("三星堆文物修复模型");
    //QString qstr = QString::fromLocal8Bit("三星堆文物修复模型");
    this->setWindowTitle(qstr);

    QLabel *lab= new QLabel(this);
    //lab->setGeometry(20,20,extending,extending);
    lab->setGeometry(20,20,960,560);
    lab->setAlignment(Qt::AlignTop);
    lab->setText(QString::fromLocal8Bit("使用指南："
                 "<br/>"
                 "<br/>"
                 "双击软件图标进入软件，欢迎界面显示三秒后自动跳转主界面，无需用户操"
                 "<br/>"
                 "作。"
                 "<br/>"
                 "用户可以点击“上传图片”按钮上传待修复的文物图；直接输入待修复文物"
                 "<br/>"
                 "图的本地地址也是可行的。"
                 "<br/>"
                 "点击“点击修复”按钮，破损文物图片被修复并展示在右侧虚线框中。同时"
                 "<br/>"
                 "保存在本地，用户可以点击“打开目录”按钮查看修复后图片。"
                 "<br/>"
                 "点击菜单栏帮助按钮，可以联系我们。"));
    lab->setStyleSheet("QLabel{font:20px}");
}
