#include "mnistreader.h"
int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void MNISTReader(std::ifstream& file,std::vector< std::vector<float> >& images, int& magic, int& size, int& rows, int& columns){
    file.read((char*) &magic, sizeof(int));
    magic = ReverseInt(magic);
    file.read((char*) &size, sizeof(int));
    size = ReverseInt(size);
    file.read((char*) &rows, sizeof(int));
    rows = ReverseInt(rows);
    file.read((char*) &columns, sizeof(int));
    columns = ReverseInt(columns);
    images.resize(size, std::vector<float> (rows * columns));
    for (size_t i = 0; i < size; ++i)
        for (size_t j = 0; j < rows * columns; ++j){
            unsigned char x;
            file.read((char*) &x, sizeof(x));
            images[i][j] = ((float) x) / 255;
        }
}
void MNISTReader(std::ifstream& file, std::vector<float>& labels, int& magic, int& size){
    file.read((char*) &magic, sizeof(int));
    magic = ReverseInt(magic);
    file.read((char*) &size, sizeof(int));
    size = ReverseInt(size);
    labels.resize(size);
    for (size_t i = 0; i < size; ++i){
        char x;
        file.read((char*) &x, sizeof(x));
        labels[i] = ((float) x);
    }
}
