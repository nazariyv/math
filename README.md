# Docs

## TODOs

- write a readme with math for each of these files

line is described by a point and a vector
denote the point by $p_0$ and the vector by $\vec{v}$
then line (call it $l$) is described by the equation $l = p_0 + tv$

denote shortest distance from origin to this line by $d(l, O)$

now recall the concept of projection

consider line $l'$ which is parallel to $l$, but it passes through origin
so it will be described by the equation $l' = 0 + tv$
now, $d(l, O) = d(l', p_0)$

if you think of $p_0$ as a vector from origin to that point,
then $d(l', p_0) = || P_{\vec{v} \perp}(p_0) ||$
i.e. perpendicular projection of $\vec{p_0}$ on $l'$ (because vector $\vec{v}$ is parallel to $l'$)
where $P_{\vec{v} \perp}(\vec{p_0}) = \vec{p_0} - P_{\vec{v} \perp}(p_0)$
where $P_{\vec{v} \perp}(\vec{p_0}) = (\vec{p_0} \cdot \vec{v} / ||\vec{v}||^2) \vec{v}$
