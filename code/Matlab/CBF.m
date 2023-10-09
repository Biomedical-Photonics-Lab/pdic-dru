%% u cut off at 2*pi*(NAc + NAo)/lam. So the 2*pi factor should be included in u.
function c = CBF(u, NAc, NAo, lam)
    r2 = 2*pi*NAc/lam;
    r1 = 2*pi*NAo/lam;
    assert(NAo >= NAc);
    
    d1 = (u.^2 + r1^2 - r2^2)./(2*u);
    d2 = (u.^2 + r2^2 - r1^2)./(2*u);
    c = r1^2*acos(d1./r1) + r2^2*acos(d2./r2) ...
        - d1.*sqrt(r1^2 - d1.^2) - d2.*sqrt(r2^2 - d2.^2);
    
    c = c/(2*pi)^2;
    c(u>r1+r2) = 0;
    c(u<r1-r2) = pi*r2^2/(2*pi)^2;
    
    %% Normalize to unity at zero
    c = c/(pi*r2^2/(2*pi)^2);
    c(u==0) = 1;
