#include<iostream>
#include<stdlib.h>
#include<cmath>
#include<mpi.h>
using namespace std;
int N=20;
int root =0;
bool boundary(int x, int y);
int main(int argc, char *argv[])
{

	MPI_Init(&argc,&argv);
	int p, rank;
	double time0=MPI_Wtime();
	int from, to;
	MPI_Comm_size( MPI_COMM_WORLD, &p );

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);	
	int i,j,x,y,k;
	int size;
	double dot =0,finres;
	size=(N-2)*(N-2);
	double *B=new double[size];
	// to make sure that the first processors get the 1 more data point than the later ones
	 if(rank<(size%p))
	{
		from=rank*size/p+rank;
		to=from+size/p+1;
	}
	else
	{
		from=rank*size/p+size%p;
		to=(rank+1)*size/p;
	}
	//initializing the values of B matrix	
	double result;
	for(i=from;i<to;i++)
	{
		result =0.0;
		x=i%(N-2)+1;
		y=i/(N-2)+1;
		double x_org = (double)x/(N-1);
		if(boundary(x+1,y))
			result+=0.0;
		if(boundary(x-1,y))
			result+=0.0;
		if(boundary(x,y+1))
			result+=sin(M_PI*x_org)*exp(-M_PI);
		if(boundary(x,y-1))
			result+=sin(M_PI*x_org);
		B[i-from]=-result;
	}
	//calculating the values of A matrix
	int *a=new int[size*size];
	int *column= new int[size*size];
	int *row=new int[size];
	row[0]=0;
	double *B_V= new double[size];
	double *P_V=new double[size];
	double *Z=new double[size];
	double *R_V=new double[size];
	for(i=0;i<size;i++)
	{
		B_V[i]=0.0;
		P_V[i]=0.0;
		Z[i]=0.0;
		R_V[i]=0.0;
	}
	k=0;
	for(i=from;i<to;i++)
	{
		a[k]=-4;
		column[k]=i;
		k++;
		if(i%(N-2)!=0)
		{
			a[k]=1;
			column[k]=i-1;
			k++;
		}
		if((i+1)%(N-2)!=0)
		{
			a[k]=1;
			column[k]=i+1;
			k++;
		}
		if(i>=N-2)
		{
			a[k]=1;
			column[k]=i-(N-2);
			k++;
		}
		if(i+(N-2)<(N-2)*(N-2))
		{
			a[k]=1;
			column[k]=i+(N-2);
			k++;
		}
		row[i-from+1]=k;
	}
	for(i=0;i<(to-from)+1;i++)
	{
		//	cout<<row[i]<<endl;
	}
	int start=0;
	int bcount[p];
	int offset[p];
	for(i=0;i<p;i++)
	{
		offset[i]=start;
		if(i<size%p)
			bcount[i]=size/p+1;
		else
			bcount[i]=size/p;	
		start+=bcount[i];
	}
	int *precol=new int[size];
	int *prerow=new int[size+1];
	double *m=new double[size];
	
	prerow[0]=0;
	for(i=offset[rank];i<(offset[rank]+bcount[rank]);i++)
	{
		m[i-offset[rank]]=0.25;
		precol[i-offset[rank]]=i;
		prerow[i+1-offset[rank]]=i+1-offset[rank];
		
	}
// All gather for matrix vector multiplication of B and M(preconditioning) to get Z. 	
	MPI_Allgatherv(&B[0],bcount[rank],MPI_DOUBLE,&P_V[0],bcount,offset,MPI_DOUBLE,MPI_COMM_WORLD);

		for(i=0;i<bcount[rank];i++)
		{  //cout<<row[i]<<endl;
			for(j=prerow[i];j<prerow[i+1];j++)
			{
			Z[i]=Z[i]+m[j]*P_V[precol[j]];
			}  
		}
// To calculate the conjugate gradient with jacobi preconditioning 
	double *R=new double[size];
	double *P=new double[size];
	double *X=new double[size];
	double glalden=0.0;
	double glalnum=0.0;
	double glarsum=0.0;
	double alpha, beta;
	//double Ro[size],Po[size];
	int it;
	for(i=0;i<bcount[rank];i++)
	{
		R[i]=B[i];
		P[i]=Z[i];
		X[i]=0.0;
	}
//	double matvec[size];
	for(it=0;it<4000;it++)
	{
		//calculating Alpha	
		//calculating the denominator
		//APk
		//vector multiplication
		MPI_Allgatherv(&P[0],bcount[rank],MPI_DOUBLE,&B_V[0],bcount,offset,MPI_DOUBLE,MPI_COMM_WORLD);
		double *matvec=new double[size];
		for(i=0;i<bcount[rank];i++)
		{
		matvec[i]=0;
		}
		for(i=0;i<bcount[rank];i++)
		{  //cout<<row[i]<<endl;
			for(j=row[i];j<row[i+1];j++)
			{
			matvec[i]=matvec[i]+a[j]*B_V[column[j]];
			}  
		}
		// calclulatin	g the dot product of the denominator
		double alden=0.0;
		for(i=0;i<bcount[rank];i++)
		{
			alden=alden+matvec[i]*P[i];
		}
		MPI_Allreduce(&alden,&glalden,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);	
	//	cout<<"alden global"<<glalden<<"  alden local "<<alden<<" rank--->"<<rank<<endl;

		double alnum=0.0;
		for(i=0;i<bcount[rank];i++)
		{
			alnum=alnum+Z[i]*R[i];
		}
		MPI_Allreduce(&alnum,&glalnum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);		
		alpha=0.0;
		alpha=glalnum/glalden;
//		cout<<alpha<<"<--alph , aldnum global "<<glalnum<<" "<<alnum<<" rank: "<<rank<<endl;	
		for(i=0;i<bcount[rank];i++)
		{
			X[i]=X[i]+alpha*P[i];
			R[i]=R[i]-alpha*matvec[i];
		}
		MPI_Allgatherv(&R[0],bcount[rank],MPI_DOUBLE,&R_V[0],bcount,offset,MPI_DOUBLE,MPI_COMM_WORLD);
		for(i=0;i<bcount[rank];i++)
		{  //cout<<row[i]<<endl;
			for(j=prerow[i];j<prerow[i+1];j++)
			{
			Z[i]=Z[i]+m[j]*R_V[precol[j]];
			}  
		}
		MPI_Allgatherv(&Z[0],bcount[rank],MPI_DOUBLE,&B_V[0],bcount,offset,MPI_DOUBLE,MPI_COMM_WORLD);
		if(rank==0)
		for(i=0;i<size;i++)
		{
		//cout<<B_V[i]<<endl;
		}
		double rsum=0;
		// finding the value of beta	
		for(i=0;i<bcount[rank];i++)
		{
			rsum=rsum+Z[i]*R[i];
		}
		MPI_Allreduce(&rsum,&glarsum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
		beta=glarsum/glalnum;
		// calculating the value of Pk
		for(i=0;i<bcount[rank];i++)
		{
			P[i]=Z[i]+beta*P[i];
		}
	}
	//if(rank==0)
	
		MPI_Allgatherv(&X[0],bcount[rank],MPI_DOUBLE,&B_V[0],bcount,offset,MPI_DOUBLE,MPI_COMM_WORLD);
		if(rank==0)
		{
		for(i=0;i<size;i++)
		{
			  cout<<B_V[i]<<endl;
		}
		
		cout<<endl;
		cout<<endl;
		double time=MPI_Wtime();
		cout<<(time-time0)<<" <----- time taken"<<endl;
		}
	MPI_Finalize();
}
bool boundary(int x, int y)
{
	if(x==0||x==N-1||y==0||y==N-1)
	{
		return true;
	}
	else 
		return false;
}
