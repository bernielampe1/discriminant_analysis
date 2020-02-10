function x = gaussIntersect(u1, s1, u2, s2)
    % tolerance for determining if std devs are equal to avoid div zero
    tol = 0.000000001;

    % check if both std devs are equal
    if abs(s1 -s2) < tol
        x = (u1 + u2) * 0.5;
        return;
    end
    
    % solve quadratic system derived via substitution
    a = s1 * s1 - s2 * s2;
    b = 2 * (u1 * s2 * s2 - u2 * s1 * s1);
    c = (u2 * u2 * s1 * s1 - u1 * u1 * s2 * s2) - 2 * s1 * s1 * s2 * s2 * (log(s1) -log(s2));
    
    x1 = (-b + sqrt(b*b - 4 * a * c)) / (2 * a);
    x2 = (-b - sqrt(b*b - 4 * a * c)) / (2 * a);
    x = x1;
end
