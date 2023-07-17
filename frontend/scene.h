#ifndef SCENE_H
#define SCENE_H

#include <QMainWindow>

class scene : public QMainWindow
{
    Q_OBJECT
public:
    explicit scene(QWidget *parent = nullptr);
    void paintEvent(QPaintEvent *event);

    // 用于存储图像路径
    std::string img_path;
private:

signals:

public slots:
};

#endif // SCENE_H
