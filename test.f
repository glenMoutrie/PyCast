c Test program hello
        

program hello
    double precision test(10), ave(10)
    integer n, total, testTwo(10)
    
    n = 10
    
    do 10 i = 1,n
        test(i) = i
        ave(i) = 0
        print *, test(i)

    10 continue
    
    print *, n
    
    !call stlma(test, n, 3, ave)
    call stlfts(test, n, 2, ave)
    
    print *, n
    
    do 11 j = 1,n
        print *, "moving average: ", test(j), ave(j)
    11 continue
    
    print *, "test"
    
end program hello

subroutine stlfts(x,n,np,trend,work)
      integer n, np
      double precision x(n), trend(n), work(n)

      call stlma(x,    n,      np, trend)
      call stlma(trend,n-np+1, np, work)
      call stlma(work, n-2*np+2,3, trend)
      return
end

subroutine stlma(x, n, len, ave)

      integer n, len
      double precision x(n), ave(n)
      
      double precision flen, v
      integer i, j, k, m, newn
      
      
      print *, "inside stlma", n  
      
      newn = n-len+1
      flen = dble(len)
      v = 0.d0
      do 3 i = 1,len
         v = v+x(i)
 3    continue
      ave(1) = v/flen
      if(newn .gt. 1) then
         k = len
         m = 0
         do 7 j = 2, newn
            k = k+1
            m = m+1
            v = v-x(m)+x(k)
            ave(j) = v/flen
 7       continue
      endif
      return
end


