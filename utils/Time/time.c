#include <time.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>

clock_t start_time, end_time;
double total_t;
long mtime, seconds, useconds;

struct timeval start, end;

/**
 * Method to start the timer.
 */
void startTimer()
{
    start_time = clock();
    gettimeofday(&start, NULL);
}

/**
 * Method to stop the timer.
 */
void endTimer()
{
    end_time = clock();
    gettimeofday(&end, NULL);
}

/**
 * Method to get the time elapsed on CPU.
 */
double getTime()
{
    total_t = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    return total_t;
}

/**
 *Get real time
 */
double getwallClockTime()
{
    seconds = end.tv_sec - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;
    double duration;
    duration = seconds + useconds / 1000000.0;
    return duration;
}