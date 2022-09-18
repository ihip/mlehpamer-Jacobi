#ifndef _GET_TIME_
#define _GET_TIME_

#ifdef __cplusplus
// when included in C++ file, let compiler know these are C functions
extern "C"
{
#endif

    void startTimer();
    void endTimer();
    double getTime();
    double getwallClockTime();

#ifdef __cplusplus
}
#endif

#endif /* _GET_TIME_*/