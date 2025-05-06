theory TEMP
imports Complex_Main
begin
theorem inequality_test1:
  fixes a b :: real
  assumes h0: "0 < a"
    and h1: "0 < b"
    and h2: "a + b = 1"
  shows "a * b \<le> 1/4"
proof-
(*Since a>0, b>0, we know that (a+b)/2 >= \sqrt{ab}.*)
have h3: "(a+b)/2 >= sqrt (a*b)" using h0 h1 sorry
(*Then, with a+b=1, we know that \sqrt{ab}<=1/2.*)
have h4: "sqrt (a*b) <= 1/2" using h2 sorry
(*Therefore, we know that ab <= 1/4.*)
have "a*b <= 1/4" using h3 h4 sorry
(*QED*)
then show ?thesis  by blast
qed

end