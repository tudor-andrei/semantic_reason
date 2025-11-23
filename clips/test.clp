;;;; (este un-actor-grozav $?nume-complet)
;;;; (joaca-la un-actor-grozav $?productie)

(defrule regula-test
	(este un-actor-grozav $?nume-complet)
	(joaca-la un-actor-grozav $?productie)
	=>
	(printout t $?nume-complet " joaca la " $?productie crlf)
)

(deffacts fapte-test
   (este un-actor-grozav Florin Piersic)
   (joaca-la un-actor-grozav Studioul Hollywood)
)
