# Reflection on AI Paradoxes

This homework showed me a practical paradox: an AI system can become better at
checking facts and still be fooled by the document it is checking. My verifier
rejected the first attack, which used hidden instructions. It then accepted a
PDF whose visible job title differed from its extracted text. The model compared
the text it received correctly, but that text did not represent what a person
would see on the page. A high score therefore did not establish that the whole
document was trustworthy.

This connects to the automation paradox. As more routine review is automated,
people may inspect fewer cases, even though unusual failures still require
human judgment. It also makes the productivity paradox more concrete for me:
adding an AI tool does not automatically improve a business process. The data,
document handling, review procedures and accountability all matter. In financial
technology, faster KYC or document checking would be useful, but an error could
affect a customer's access to a service. I want to learn how to evaluate the
whole workflow, including what information reaches the model, rather than judge
it only by a confidence score. I would pair automated checks with audit records
and a clear way to escalate uncertain cases for human review.
