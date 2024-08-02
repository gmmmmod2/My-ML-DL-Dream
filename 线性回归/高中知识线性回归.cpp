#include <iostream>
#include <vector>
#include <numeric>
using namespace std;

struct House {
    double area;
    double price;
};

// 计算向量的平均值
double mean(const vector<double>& v) {
    double sum = accumulate(v.begin(), v.end(), 0.0);
    return sum / v.size();
}

// 线性回归函数
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
    // 示例数据：面积（平方米）和价格（单位）
    std::vector<House> houses = {
        {90, 500000},
        {100, 550000},
        {130, 700000},
        {150, 800000}
    };

    double a, b;
    linear_regression(houses, a, b);

    cout << "线性回归方程: y = " << a << "x + " << b << std::endl;

    // 预测面积为140平方米的房屋价格
    double area_to_predict = 140;
    double predicted_price = a * area_to_predict + b;

    cout << "预测面积为 " << area_to_predict << " 平方米的房屋价格为: " << predicted_price << std::endl;

    return 0;
}
