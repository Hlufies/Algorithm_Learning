# 快速排序 quick_sort
```
using namespace std;

const int N = 100010;

int q[N];

void quick_sort(int A[], int l, int r)
{
    if(l >= r) return;
    int i = l-1, j = r+1, x = A[l+r >> 1];
    while(i < j)
    {
        do i++; while(A[i] < x);
        do j--; while(A[j] > x);
        if(i < j) swap(A[i], A[j]);
    }
    quick_sort(A, l, j), quick_sort(A, j+1, r);
}

int main()
{
    int n;
    cin >> n;
    for(int i = 0; i < n; i++) cin >> q[i];
    quick_sort(q, 0, n-1);
    for(int i = 0; i < n; i++) cout << q[i] << ' ';
    return 0;
}

```
```
第k个数
void quick_sort(int A[], int l, int r, int k){
    if (l >= r) return A[l];
    int i = l - 1, j = r+1, x = A[l+r >> 1];
    
    while(i < j){
        do i++; while(A[i] < x);
        do j--; while(A[j] > x);
        if (i < j) swap(A[i], A[j]);
    }
    if return (j-l+1 >= k)quick_sort(A, l, j, j-l+1-k);
    else return quick_sort(A, j+1, r, k+l-j-1);
}
```
