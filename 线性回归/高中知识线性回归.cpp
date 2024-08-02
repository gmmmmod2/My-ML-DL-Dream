#include <iostream>
#include <vector>
#include <numeric>
using namespace std;

struct House {
    double area;
    double price;
};

// ����������ƽ��ֵ
double mean(const vector<double>& v) {
    double sum = accumulate(v.begin(), v.end(), 0.0);
    return sum / v.size();
}

// ���Իع麯��
void linear_regression(const vector<House>& houses, double& a, double& b) {
    vector<double> areas, prices;

    for (const auto& house : houses) {
        areas.push_back(house.area);
        prices.push_back(house.price);
    }

    double mean_area = mean(areas);
    double mean_price = mean(prices);

    double numerator = 0.0;
    double denominator = 0.0;

    for (size_t i = 0; i < houses.size(); ++i) {
        numerator += (areas[i] - mean_area) * (prices[i] - mean_price);
        denominator += (areas[i] - mean_area) * (areas[i] - mean_area);
    }

    a = numerator / denominator;
    b = mean_price - a * mean_area;
}

int main() {
    // ʾ�����ݣ������ƽ���ף��ͼ۸񣨵�λ��
    std::vector<House> houses = {
        {90, 500000},
        {100, 550000},
        {130, 700000},
        {150, 800000}
    };

    double a, b;
    linear_regression(houses, a, b);

    cout << "���Իع鷽��: y = " << a << "x + " << b << std::endl;

    // Ԥ�����Ϊ140ƽ���׵ķ��ݼ۸�
    double area_to_predict = 140;
    double predicted_price = a * area_to_predict + b;

    cout << "Ԥ�����Ϊ " << area_to_predict << " ƽ���׵ķ��ݼ۸�Ϊ: " << predicted_price << std::endl;

    return 0;
}
