---
# Display name
name: CamemBERT

# Username (this should match the folder name)
authors:
- admin

# Is this the primary user of the site?
superuser: true

# Role/position
role:  A Tasty French Language Model

# Organizations/Affiliations
organizations:
- name: Facebook AI Research
  url: "https://ai.facebook.com"
- name: Inria
  url: "https://www.inria.fr/en/"
- name: ALMAnaCH
  url: "https://team.inria.fr/almanach/"

# Short bio (displayed in user profile at end of posts)
bio: A Tasty French Language Model trained by Louis Martin, Benjamin Muller, Pedro Javier Ortiz Suárez, Yoann Dupont, Laurent Romary, Éric Villemonte de la Clergerie, Djame Seddah and Benoît Sagot.

# Social/Academic Networking
# For available icons, see: https://sourcethemes.com/academic/docs/widgets/#icons
#   For an email link, use "fas" icon pack, "envelope" icon, and a link in the
#   form "mailto:your-email@example.com" or "#contact" for contact widget.
social:
- icon: envelope
  icon_pack: fas
  link: '#contact'  # For a direct email link, use "mailto:test@example.org".
- icon: twitter
  icon_pack: fab
  link: https://twitter.com/InriaParisNLP
# - icon: google-scholar
#   icon_pack: ai
#   link: https://scholar.google.co.uk/citations?user=sIwtMXoAAAAJ
# - icon: github
#   icon_pack: fab
#   link: https://github.com/gcushen
# Link to a PDF of your resume/CV from the About widget.
# To enable, copy your resume/CV to `static/files/cv.pdf` and uncomment the lines below.  
# - icon: cv
#   icon_pack: ai
#   link: files/cv.pdf

# Enter email to display Gravatar (if Gravatar enabled in Config)
email: ""
  
# Organizational groups that you belong to (for People widget)
#   Set this to `[]` or comment out if you are not using People widget.  
user_groups:
- Researchers
- Visitors
---

CamemBERT is a state-of-the-art language model for French based on the [RoBERTa architecture](https://ai.facebook.com/blog/roberta-an-optimized-method-for-pretraining-self-supervised-nlp-systems/) pretrained on the French subcorpus of the newly available multilingual corpus [OSCAR](https://traces1.inria.fr/oscar/).

We evaluate CamemBERT in four different downstream tasks for French: part-of-speech (POS) tagging, dependency parsing, named entity recognition (NER) and natural language inference (NLI); improving the state of the art for most tasks over previous monolingual and multilingual approaches, which confirms the effectiveness of large pretrained language models for French.

CamemBERT was trained and evaluated by [Louis Martin](https://github.com/louismartin), [Benjamin Muller](https://benjamin-mlr.github.io), [Pedro Javier Ortiz Suárez](https://pjortiz.com), [Yoann Dupont](https://github.com/YoannDupont), [Laurent Romary](https://cv.archives-ouvertes.fr/laurentromary), [Éric Villemonte de la Clergerie](http://alpage.inria.fr/~clerger/), [Djamé Seddah](http://pauillac.inria.fr/~seddah/) and [Benoît Sagot](http://alpage.inria.fr/~sagot/).
