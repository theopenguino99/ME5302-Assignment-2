subroutine Solp
    
    use global_var
    implicit none
    
    integer :: i,j
    real*8 :: beta,bp
    
    beta = 0.4d0
    
    bp = -2.d0/(dx**2.d0)-2.d0/(dy**2.d0)
    
    do j = 2,jmax-1
        do i = 2,imax-1
            rp(i,j)=vor(i,j)-(p(i+1,j)-2.d0*p(i,j)+p(i-1,j))/(dx**2.d0)-(p(i,j+1)-2.d0*p(i,j)+p(i,j-1))/(dy**2.d0)
        end do
    end do

    do j = 2,jmax-1
        do i = 2,imax-1
            p(i,j)=p(i,j)+beta*rp(i,j)/bp
        end do
    end do

    return
end subroutine Solp