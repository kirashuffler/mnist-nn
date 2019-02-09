#ifndef MNISTREADER_H
#define MNISTREADER_H
#include <fstream>
#include <vector>
int ReverseInt(int i);
void MNISTReader(std::ifstream& stream,std::vector< std::vector<float> >& images, int& magic, int& size, int& rows, int& columns);
void MNISTReader(std::ifstream& file, std::vector<float>& labels, int& magic, int& size);
#endif // MNISTREADER_H
