#include <stdio.h>

int arrow (int num, int power, int arrownum) {
    int answer = num;
    if (arrownum == 0)
        return num * power;
    for (int i = 1; i < power; i++)
        answer = arrow(num, answer, arrownum - 1);
    return answer;
}

int main(){
  int num, power, arrownum;

  num = 2;
  power = 2;
  arrownum = 1;
  int result = arrow(num, power, arrownum);
  printf("/n");
  printf(result);

  return 0;
}

