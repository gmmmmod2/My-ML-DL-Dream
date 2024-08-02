#include<iostream>
#include<vector>
#include<math.h>
#include<iomanip>
#include<Windows.h>
// 3 is the number of elements in the array
const int m = 3;
using namespace std;

//record the cost of each iteration and parameter
double* Cost_history = new double[100000];
double* P_history = new double[100000];


// Squared error cost function
double compute_cost(double x[], double y[],double a,double b) {
	
	double total_cost=0;
	double cost=0;
	double f_ab=0;

	for (int i = 0; i < m;i++) {
		f_ab = a * x[i] + b;
		cost += (f_ab - y[i])*(f_ab-y[i]);
	}
	total_cost = cost / (2.0 * m);

	return total_cost;
}

double compute_gradient_a(double x[], double y[],double a,double b) {

	double f_ab_a = 0;
	double f_ab_i = 0;

	for (int i = 0; i < m; i++) {
		f_ab_i = (a * x[i] + b - y[i])*x[i];
		f_ab_a += f_ab_i;
	}
	return f_ab_a / m;
}

double compute_gradient_b(double x[], double y[], double a, double b) {
	double f_ab_b = 0;
	double f_ab_i = 0;

	for (int i = 0; i < m; i++) {
		f_ab_i = a * x[i] + b - y[i];
		f_ab_b += f_ab_i;
	}
	return f_ab_b / m;
}

void linearReression(double x[], double y[],double alpha,double a_in, double b_in,int iter_nums) {
	//initiazation
	double a = a_in;
	double b = b_in;

	int i=0;
	while (i < iter_nums) {
		double temp_a = compute_gradient_a(x, y, a, b);
		double temp_b = compute_gradient_b(x, y, a, b);
		a = a-alpha*temp_a;
		b = b-alpha*temp_b;

		double cost=0;
		if (i < 100000) {
			cost = compute_cost(x, y, a, b);
			//save data during iteration 
			Cost_history[i] = cost;
			P_history[i] = (int)((a * 10000 + 0.5)*10) + (int)(b * 10000 + 0.5) / 10000.0;
		}
		if (i % 1000 == 0) cout << fixed << setprecision(5)<<"Iteration: " << setw(5) << i << "  Cost: " << setw(5) << cost << "  a: " << setw(5) << a << "  b: " << setw(5) << b<<endl;

		i++;
	}
	cout << "(a,b)found that: a=" << a << "  b=" << b<<endl;
}

int main() {
	// record start time
	LARGE_INTEGER t1, t2, tc;
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&t1);

    // load data set
	double train_x[3] = { 0.0,1.0,2.0 };
	double train_y[3] = { 0.0,2.0,3.0 };

	double a_in = 0;
	double b_in = 0;
	double alpha = 0.01;
	linearReression(train_x, train_y, alpha, a_in, b_in, 10000);

	// remember delete
	delete []Cost_history;
	delete []P_history;

	// record end time
	QueryPerformanceCounter(&t2);
	double elapsedTime = (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart;
	cout << "time = " << elapsedTime << endl;  //输出时间（单位：ｓ）
	return 0;
}