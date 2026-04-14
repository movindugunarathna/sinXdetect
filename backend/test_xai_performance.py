#!/usr/bin/env python3
"""Test XAI techniques (LIME vs SHAP) performance on multiple Sinhala texts."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
for search_path in (CURRENT_DIR, ROOT_DIR):
    search_path_str = str(search_path)
    if search_path_str not in sys.path:
        sys.path.insert(0, search_path_str)

from compare_xai_explanations import run_comparison

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Test texts
TEST_TEXTS = [
    {
        "name": "iran_geopolitics",
        "topic": "Geopolitics (Iran-US conflict)",
        "text": """අද ලෝකයේ ඉරානය සම්බන්ධ යුද්ධය දේශපාලනික වශයෙන් ඉතා සංකීර්ණ ගැටළුවක් ලෙස සැලකේ. මෙය සරලව යුධයකට පමණක් සීමා නොවී, ලෝක බලවතුන් අතර බල තරඟයක් ලෙසද දැක්විය හැක. එක් පැත්තකින් ඇමරිකාව සහ ඉස්රායලය ඉරානයේ න්‍යෂ්ටික බලය හා ආරක්ෂක තර්ජනයක් ගැන සැලකිලිමත් වන අතර, අනෙක් පැත්තෙන් ඉරානය තම ආරක්ෂාව සහ ස්වාධීනත්වය රැක ගැනීමට උත්සාහ කරයි. දේශපාලන විශ්ලේෂකයින් අනුව, මෙම යුද්ධය මැද පෙරදිග ප්‍රදේශයේ බල සමතුලිතතාවය වෙනස් කිරීමට හේතු වන එකක් ලෙස පෙනේ. සමහර රටවල් ඇමරිකාවට සහාය දක්වමින් ක්‍රියා කරන අතර, වෙනත් රටවල් මෙම ගැටළුවට සාමකාමී විසඳුමක් අවශ්‍ය බව ප්‍රකාශ කරයි.

තවද, මෙම යුද්ධය ආර්ථික හා බලශක්ති ක්ෂේත්‍රයටද විශාල බලපෑමක් ඇති කරමින් ලෝකයේ තෙල් මිල ඉහළ යාම සහ ආහාර ආරක්ෂාවට තර්ජනයක් ඇති කරයි . එමඟින් යුද්ධය පමණක් නොව, ලෝක දේශපාලනයම අස්ථිර වීමට හේතු වේ. අවසානே, ඉරානයේ යුද්ධය දේශපාලනිකව බලවතුන්ගේ ආරක්ෂාව, බලය සහ බලපෑම සම්බන්ධ සටනක් ලෙස හඳුනාගත හැක. එය විසඳීමට හොඳම මාර්ගය සාකච්ඡා සහ රාජ්‍යතාන්ත්‍රික උත්සාහයන් බව බොහෝ දෙනා විශ්වාස කරති."""
    },
    {
        "name": "digital_education",
        "topic": "Digital Education",
        "text": """ඩිජිටල් තාක්ෂණය වේගයෙන් වර්ධනය වන අද කාලයේ, අධ්‍යාපන ක්ෂේත්‍රය තුළද විශාල වෙනසක් සිදුවී ඇත. ඩිජිටල් අධ්‍යාපනය යනු පරිගණක, අන්තර්ජාලය හා අනෙකුත් තාක්ෂණික මෙවලම් භාවිතා කරමින් සිදු කරන ඉගෙනුම් ක්‍රමයකි. මෙය ශිෂ්‍යයන්ට ඉගෙනීම පහසු, වේගවත් සහ ආකර්ෂණීය කරයි.

ඩිජිටල් අධ්‍යාපනය මගින් ලෝකයේ ඕනෑම ස්ථානයක සිට දැනුම ලබා ගැනීමට හැකි වීම එහි විශේෂ වාසියකි. මාර්ගගත පන්ති, වීඩියෝ පාඩම් සහ අන්තර්ක්‍රියාකාරී යෙදුම් මඟින් ශිෂ්‍යයන්ට තමන්ට අවශ්‍ය වේලාවේ ඉගෙනීමට අවස්ථාව ලැබේ. එමෙන්ම, ගුරුවරුන්ටද නවීන ඉගැන්වීම් ක්‍රම භාවිතා කරමින් පාඩම් ඉදිරිපත් කිරීමට හැකිවේ.

කෙසේ වෙතත්, ඩිජිටල් අධ්‍යාපනයේ අවාසිද ඇත. සියලුම ශිෂ්‍යයන්ට තාක්ෂණික උපකරණ හා අන්තර්ජාල පහසුකම් නොමැතිවීම විශාල ගැටළුවකි. එමෙන්ම, අධික ලෙස තිර භාවිතය සෞඛ්‍යයටද අහිතකර විය හැක.

ඒ අනුව, ඩිජිටල් අධ්‍යාපනය නිවැරදිව හා සමානව භාවිතා කළහොත් එය අධ්‍යාපන ක්ෂේත්‍රයේ විප්ලවයක් ඇති කළ හැක. එබැවින්, සියලු දෙනාටම මෙම අවස්ථා ලබා දීම සඳහා සුදුසු සැලසුම් ක්‍රියාත්මක කිරීම අත්‍යවශ්‍ය වේ."""
    },
    {
        "name": "assessment_methods",
        "topic": "Assessment Methods in Education",
        "text": """ඇගයීමක මූලික අරමුණ වන්නේ ගුරුවරයා ඉගැන්වූ කරුණු සිසුවා සත්‍යයෙන්ම ඉගෙනගෙන ඇතිද යන්න පරීක්ෂා කිරීමයි. මෙම ඇගයීම් හෝ පරීක්ෂණ විවිධ ආකාරයන්ගෙන් පවතින අතර, බහු වරණ, කෙටි පිළිතුරු, රචනා ප්‍රශ්න සහ සිතියම් සලකුණු කිරීම වැනි ක්‍රම ඒ අතරට අයත් වේ. විද්වත් මතය අනුව, පසුගිය සියවස තුළ ඇගයීම් ක්‍රමවල විශාල වෙනසක් සිදුවී නොමැති අතර, ඒවා බොහෝවිට එකම ආකාරයකට පවතිනවා ලෙස පෙනේ. සමහර අවස්ථාවල එවැනි ඇගයීම් ඒකාකාරී වීමත්, වෙනත් අවස්ථාවල සිසුන්ගේ මනෝභාවයට අහිතකර ලෙස බලපාන ස්වභාවයක් ද ගනී. එබැවින්, කෘත්‍රිම බුද්ධිය භාවිතයෙන් මෙම ක්‍රම නවීකරණය කිරීම වැදගත් වේ.

මෙම ගැටළුවට විසඳුම් සෙවීම සඳහා හාවර්ඩ් විශ්වවිද්‍යාලය "Reach Every Reader" යන ව්‍යාපෘතිය ක්‍රියාත්මක කරමින් සිටී. එසේම, ෆ්ලොරීඩා ප්‍රාන්ත විශ්වවිද්‍යාලය සහ මැසචුසෙට්ස් තාක්ෂණායතනය (MIT) දරුවන්ට සහ දෙමාපියන්ට එක්ව භාවිතා කළ හැකි අධ්‍යාපනික ක්‍රීඩා සංවර්ධනය කරයි. සත්‍ය ජීවිතයේ චරිත හා සිදුවීම් පදනම් කරගත් role-play වර්ගයේ ක්‍රීඩාද මෙයට ඇතුළත් වේ. මෙම ක්‍රම මගින් ප්‍රාථමික ශ්‍රේණිවල දරුවන්ගේ ඉගෙනුම් හැකියාව වැඩිදියුණු කිරීමටත්, ඔවුන්ගේ අධ්‍යාපන මට්ටම පිළිබඳ නිවැරදි තක්සේරුවක් ලබා ගැනීමටත් හැකි වේ."""
    },    {
        "name": "democracy",
        "topic": "Democracy and Citizenship",
        "text": """ප්‍රජාතන්ත්‍රවාදය ගැන කතා කරන විට අපට බොහෝ විට මතක් වන්නේ මැතිවරණ සහ දේශපාලන පක්ෂයි. නමුත් සැබෑ ප්‍රජාතන්ත්‍රවාදය යනු වසර කිහිපයකට වරක් ඡන්ද පත්‍රිකාවක සලකුණක් යෙදීමෙන් පමණක් අවසන් වන්නක් නොවේ. එය රටක සාමාන්‍ය ජනතාව සතු බලය සහ එම බලය පාලකයන් කෙරෙහි බලපෑම් කරන ආකාරය පිළිබඳ ක්‍රියාවලියකි. අද අපේ සමාජය දෙස බැලීමේදී පෙනී යන්නේ දේශපාලනය කෙරෙහි ජනතාව තුළ යම් කලකිරීමක් පවතින බවයි. මෙයට ප්‍රධාන හේතුව දූෂණය සහ විනිවිදභාවය නොමැතිකමයි. එහෙත්, පද්ධතියක් නිවැරදි කිරීමට නම් පුරවැසියන් ලෙස අප සතු වගකීම අතිමහත්ය. තීන්දු තීරණ ගන්නා ආකාරය ගැන විමසිල්ලෙන් සිටීම, අසාධාරණයට එරෙහිව හඬ නැගීම සහ තොරතුරු දැනගැනීමේ අයිතිය නිවැරදිව භාවිත කිරීම ප්‍රජාතන්ත්‍රවාදී රටක ජීවත් වන ඕනෑම අයෙකුගේ යුතුකමකි. 

දේශපාලනඥයන් පත් කර යැවීමෙන් පසු අපේ වගකීම අවසන් යැයි සිතීම විශාල වැරදීමකි. රටක සැබෑ වෙනසක් සිදුවන්නේ නීතියේ ආධිපත්‍යය සුරැකෙන විට සහ පුරවැසියා ක්‍රියාකාරීව පාලන තන්ත්‍රයට සම්බන්ධ වන විටය. සැබෑ ප්‍රජාතන්ත්‍රවාදය මල්ඵල දරන්නේ පාලකයන්ට වගකීමක් දැනෙන විටත්, පාලිතයන්ට තම අයිතිවාසිකම් ගැන අවබෝධයක් ඇති විටත් පමණි. සංවර්ධිත සහ සාධාරණ සමාජයක් ගොඩනැගීමට නම් අප හුදෙක් ප්‍රේක්ෂකයන් නොවී, රටේ දේශපාලන ගමන්මග තීරණය කරන ක්‍රියාකාරී කොටස්කරුවන් විය යුතුය."""
    },
    {
        "name": "environmental_activist",
        "topic": "Environmental Conservation",
        "text": """පරිසර නියමුවා යනු හුදෙක් පරිසරයට ආදරය කරන්නෙකු පමණක් නොව, සොබාදහමේම වෙන් කළ නොහැකි කොටසකි. ගසට පොත්ත මෙන් පරිසරය හා දැඩි ලෙස බැඳී සිටින ඔහුට, වර්තමානයේ සිදුවන ශීඝ්‍ර පරිසර විනාශය හමුවේ විශාල වගකීමක් පැවරී ඇත.

සැබෑ පරිසර නියමුවෙකු සරල ජීවන රටාවකට හුරු විය යුතුය. පරිසරයට හිතකාමී වෙමින් පොලිතීන් සහ ප්ලාස්ටික් භාවිතය අවම කිරීමත්, මවක තම දරුවාට දක්වන සෙනෙහසින් යුතුව සොබාදහම රැකබලා ගැනීමත් ඔහුගේ සිරිතයි. ඔහු ගස්වැල්වල සමීපතම මිතුරෙකු විය යුතු අතර, අධ්‍යාත්මිකව මෙන්ම ප්‍රායෝගිකව ද පරිසරයේ පවිත්‍රතාව අගය කළ යුතුය. තමා පරිසරය පිරිසිදුව තබා ගන්නවා සේම, අන් අයව ද ඒ සඳහා දිරිමත් කිරීම මෙහිදී අතිශය වැදගත් වේ.

අපද්‍රව්‍ය බැහැර කිරීම අවම කරමින් පරිසර සංරක්ෂණය තම ජීවිතයේ ප්‍රධාන අරමුණ කරගත යුතු පරිසර නියමුවා, ඒ සඳහා එන ඕනෑම බාධාවකට නොබියව මුහුණ දීමට තරම් ශක්තිමත් විය යුතුය. ශාක හා සතුන් ඇතුළු සමස්ත ජීව පද්ධතියටම කරුණාව සහ දයාව දැක්වීම ඔහුගේ ලක්ෂණයකි. අභිමානවත් පරිසර නියමුවෙකු ලෙස, මගේ මව්බිමේ සොබාදහම රැක ගැනීමට වගකීමෙන් බැඳී සිටීම මගේ එකම අධිෂ්ඨානයයි."""
    },
    {
        "name": "artificial_intelligence",
        "topic": "Artificial Intelligence and Society",
        "text": """කෘතිම බුද්ධිය හෙවත් AI (Artificial Intelligence) යනු වර්තමාන ලෝකයේ තාක්ෂණික විප්ලවයේ ප්‍රධානතම ගම්‍ය බලවේගයයි. මිනිස් මොළයේ ක්‍රියාකාරිත්වය අනුකරණය කරමින් දත්ත විශ්ලේෂණය කිරීමට, තීරණ ගැනීමට සහ ගැටලු විසඳීමට හැකි පරිගණක පද්ධති නිර්මාණය කිරීම මෙහි මූලික අරමුණයි. පසුගිය දශක කිහිපය තුළ මෙම ක්ෂේත්‍රය අත්කරගෙන ඇති දියුණුව අතිමහත්ය.

වර්තමානයේ AI තාක්ෂණය සෞඛ්‍ය, අධ්‍යාපන, ප්‍රවාහන සහ මූල්‍ය වැනි විවිධ ක්ෂේත්‍ර කරා වේගයෙන් ව්‍යාප්ත වී ඇත. උදාහරණයක් ලෙස, වෛද්‍ය ක්ෂේත්‍රයේදී රෝග විනිශ්චය වඩාත් නිවැරදිව සිදු කිරීමටත්, ස්වයංක්‍රීය රථවාහන (Self-driving cars) මගින් ප්‍රවාහන පද්ධතිය කාර්යක්ෂම කිරීමටත් AI දායක වේ. එමෙන්ම, Generative AI හරහා මිනිසුන් මෙන් නිර්මාණාත්මක ලිපි ලිවීමට, රූප නිර්මාණය කිරීමට සහ සංගීතය නිපදවීමට පවා දැන් හැකියාව ලැබී ඇත.

කෙසේ වෙතත්, මෙම දියුණුවත් සමඟම විවිධ අභියෝග ද මතු වී තිබේ. රැකියා අහිමි වීමේ බිය, දත්ත රහස්‍යභාවය පිළිබඳ ගැටලු සහ සදාචාරාත්මක සීමාවන් ඒ අතර ප්‍රධාන වේ. තාක්ෂණය දියුණු වන තරමටම, එය මිනිසාගේ පාලනය යටතේ සහ පොදු යහපත උදෙසා භාවිතා කිරීම සහතික කිරීම අත්‍යවශ්‍ය වේ.

අවසාන වශයෙන්, කෘතිම බුද්ධිය යනු මානව ශිෂ්ටාචාරය නව මාවතකට යොමු කරන ප්‍රබල මෙවලමකි. එහි වාසි උපරිම කර ගනිමින් අභියෝග කළමනාකරණය කර ගැනීම අනාගත ලෝකයේ සාර්ථකත්වයට මඟ පාදනු ඇත."""
    },
    {
        "name": "trump_iran_crisis",
        "topic": "Trump's Iran War Commentary",
        "text": """ඇමරිකානු ජනාධිපති ඩොනල්ඩ් ට්‍රම්ප් සඳුදා ප්‍රකාශ කළේ යුද්ධය "ඉතා ඉක්මණින්" නිමා වනු ඇති බවයි. මෙම ප්‍රකාශයෙන් පසු තෙල් මිල පහළ වැටුණි. ට්‍රම්ප් තවද අනතුරු ඇඟවූයේ ඉරානය ජාත්‍යන්තර තෙල් ප්‍රවාහන මාර්ග අවහිර කළහොත් එයට එරෙහිව 'මරණය, ගින්න හා කෝපය' ඇති කරන බවයි. විශේෂයෙන්ම හෝර්මුස් සමුද්‍ර සන්ධියේ නැව් ගමන් මග අවහිර කිරීම සම්බන්ධයෙන් ජනාධිපතිවරයා මෙම අනතුරු ඇඟවීම් කළේය.

ට්‍රම්ප් නිවේදනයක් නිකුත් කරමින් කීවේ "හෝර්මුස් සමුද්‍ර සන්ධියේ තෙල් ප්‍රවාහනයට බාධා කිරීමට ඉරානය පියවර ගන්නේ නම්, මෙතෙක් සිදු කළ ප්‍රහාරයට වඩා විස්සක් ගුණයකින් දරුණු ප්‍රහාරයක් ඇමරිකා එක්සත් ජනපදය එල්ල කරනු ඇත" යනුවෙනි. ලෝක තෙල් නිෂ්පාදනයෙන් හෝ ප්‍රවාහනයෙන් 20%ක් පමණ මෙම හෝර්මුස් සන්ධිය හරහා ගමන් කරයි. යුද්ධය හේතුවෙන් මුහුදු ප්‍රවාහනය සැලකිය යුතු ලෙස අඩු වී ඇති අතර එහි ප්‍රතිඵලයක් ලෙස ලෝක තෙල් මිල ඉහළ ගොස් තිබේ."""
    },
    {
        "name": "examination_assessment",
        "topic": "Examination and Assessment Methods",
        "text": """විභාග ඇගයීම් ක්‍රමය
අපේ රටේ ළමයි කොච්චර දක්ෂයි, කොච්චර හැකියාවන් තියෙනවාද කියලා හිතන්නකෝ. ඒත් ඒ හැම දෙයක්ම මනින්නේ කඩදාසි කෑල්ලකින්, විභාගයකින්. ඒ හිතන කොට ටිකක් දුකයි.

අද ශ්‍රී ලංකාවේ අධ්‍යාපන ක්‍රමය ගොඩනැගිලා තියෙන්නේ ප්‍රධාන වශයෙන් විභාග කේන්ද්‍ර කරගෙනයි. A/L, O/L, ශිෂ්‍යත්වය — මේ විභාගවලට ළමයි ජීවිත කාලෙම බය වෙනවා. ඇතැම් දෙමාපියන් ළමයාගේ ප්‍රශ්නයකට "ඒක ගෑව ද?" කියලා අහනවා, "ඒකෙන් ඔයාට මොනවා ඉගෙනගන්නද?" කියලා නෙවෙයි. ඒ තරම් ලකුණු මූලික වෙලා තියෙනවා.

ඇත්තටම විභාගය නරකයි කියන්නේ නෙවෙයි. ඒකත් ඇගයීමේ එක් ක්‍රමයක්. ඒත් ගැටළුව තියෙන්නේ ඒකම එකම ක්‍රමය වෙලා තියෙනකොට. චිත්‍ර ඇඳීමේ දක්ෂ ළමයා, ක්‍රීඩාවලින් දිනාගත්ත ළමයා, මිනිස්සු හසුරුවන්නේ කොහොමද කියලා දන්න ළමයා — ඒ හැමෝටම "F" කියන හුය ගහනවා, ගණිත පිළිතුරු ලිව්වේ නැතිනම්.

ඉදිරියේදී අපිට ඕනෑ ක්‍රමාත්මකව වෙනස් කිරීමක්. ව්‍යාපෘති ඇගයීම, නිරාවරණ, නිර්මාණශීලී වැඩකටයුතු — මේවාත් ශිෂ්‍යයාගේ හැකියාව මැනිය හැකි ක්‍රම. ළමයෙකු ලකුණකට අඩු නෙවෙයි, ඊට වඩා ගොඩක් දේ."""
    },
    {
        "name": "ai_developers_future",
        "topic": "AI and Software Development",
        "text": """කලින් කාලේ කේතයක් ලිවීම කියන්නේ දවස් ගාණක් ගත වෙන වැඩක්. දැන් AI මෙවලම් ආවාට පස්සේ ඒ සියල්ල වෙනස් වෙලා. GitHub Copilot, ChatGPT වගේ දේවල් developer කෙනෙකුගේ දිනචරියාව සම්පූර්ණයෙන්ම හැඩ ගස්සලා.

ඒත් ඇත්ත ප්‍රශ්නය තමයි — AI නිසා software developer ලාට රැකියා නැති වෙයිද? බොහෝ දෙනා හිතන්නේ ඒ විදිහට. ඒත් ඇත්තටම වෙන්නේ ඊට වෙනස් දෙයක්. AI කරන්නේ කම්කරු වැඩ ගන්න එක — code ලිවීම, bugs හොයා ගන්න එක, documentation හදන එක. Developer ට ඉතිරි වෙන්නේ ඊටත් ඉහළ මට්ටමේ දේවල් — ගැටළු හඳුනා ගැනීම, නිවැරදි ප්‍රශ්නය ඇහීම, ව්‍යාපාරික තේරීම් ගැන හිතීම.

ඒ නිසා අනාගතයේ software developer කෙනෙකුට coding දෙවෙනි දෙයක් වෙයි. ප්‍රධාන කාර්යය වෙන්නේ AI ට නිවැරදි දිශාව පෙන්නීම, ඒ output ඇගයීම, සහ මිනිස් අවශ්‍යතා තේරුම් ගැනීම. ඒකට technical දැනුමත් ඕනෑ, නමුත් critical thinking සහ creativity ඊටත් වඩා ඕනෑ.

AI නිසා software field එක ඉවර වෙන්නේ නෑ. ඒ field එක grow වෙනවා — නමුත් ඒ grow එකට සූදානම් නැති කෙනාට පමණයි අවදානම. ඉගෙන ගන්නා කෙනාට AI යුගය අවස්ථාවක්."""
    },
    {
        "name": "politics_power_history",
        "topic": "Politics and Power Throughout History",
        "text": """මානවයා ජානගතව ම අනෙකා පාලනය කිරීම පිළිබඳ ව අදහසක් තිබූ සත්ත්වයෙකි. එබැවින් මානව ශිෂ්ඨාචාරයේ මුල සිටම ඔවුන්ට යම් නායකත්වයක් පිළිබඳ සහ තමන්ට අයිති වපසරියක් පිළිබඳ ව හැඟීමක් පැවතිණි. එය ගහකොළ සතාසීපාවා හෝ තමාගේම මානව වර්ගයා වෙත හෝ විහිදුණු එකක් විය. ක්‍රමික ව ජාත්‍යන්තරවාදී දේශපාලනය තෙක් විවිධ වේදිකා හරහා වර්ධනය වන්නේ මානවයාගේ එකී දේශපාලනික නැඹුරුව යි. ප්‍රථමයෙන් ම කුඩා දඩයම් කණ්ඩායම් තුළින් නායකයෙක් පත් කර ගැනීමේ සිට ගොවියුගයේ ස්ථිර ජනාවාස ඇරඹීමත් සමග ගම්නායකයෙන් පත් කර ගැනීමත්, ගම්නායකයින් එකතුවීමෙන් ප්‍රාදේශීය පාලකයන් ද, අනතුරු ව ජාතික රාජ්‍ය ද, ඉන් අනතුරු ව ජාත්‍යන්තරවාදී කණ්ඩායම් ද බිහිවීම දක්වා මෙම දේශපාලනය විහිදේ.

ඒ ඒ කාලවල්වල හඳුන්වා දුන් රාජාණ්ඩු, සමූහාණ්ඩු, ප්‍රජාතන්ත්‍රවාදී රාජ්‍ය වැනි විවිධාකාර දේශපාලන සංකල්පයන් තුළින් සංවර්ධනය වෙමින් ආ මානව දේශපාලනය වර්තමානය වන විට ඉතිහාසයේ කවරදා හෝ පැවති සංකීර්ණතම තත්ත්වයට පත් වී ඇත. මුල් කාලයේ මෙම බලය තීරණය වන්නට ඇත්තේ හුදෙක් ශාරීරික බලය මත විය යුතු ය. ඉන් අනතුරු ව විවිධ සොයාගැනීම් සමග බුද්ධිය දක්වා එය වෙනස් වන්නට ඇති අතර එයින් ද අනතුරු ව මිල මුදල් වත්කම් යනාදිය වෙතත්, වර්තමානයේ තොරතුරු වෙතත් හුවමාරු වී ඇත. අද වන විට දේශපාලන බලය ලබා ගැනීමට තත්කරන කවරෙක් වුව ද ඔහු විසින් පාලනය කිරීමට අපේක්ෂා කරනු ලබන්නාවූ ජනතාව වෙත ළඟා වන තොරතුරු සහ සන්දේශ පාලනය කිරීමටත්, ඒවා තමාට වාසිදායක අයුරින් සකසා ගැනීමටත් උනන්දු විය යුතු ය.

මිනිසාගේ ඇති බලය සහ වත්කම වෙනුවෙන් අධික කැමැත්ත නිසාවෙන් ම තවදුරටත් දේශපාලනය යන්න විවිධ සන්ධර්භයන් තුළ වර්ධනය වනු වැළකිය නොහේ. අනාගතයේ එය පෘතුවියෙන් එපිටට ද විහිද යනු ඇත."""
    },
    {
        "name": "aviation_history",
        "topic": "Aviation History and Human Achievement",
        "text": """මිහිමත ජීවය ආරම්භ වූ දා පටන්, බුද්ධිය කරපින්නා  සපැමිණි මානවයාට වසර සිය දහස් ගණනක් යනතුරුත් ජය ගැනීමට නොහැකි වූ ප්‍රධානතම අභියෝගය වූයේ ගුරුත්වාකර්ෂණය පරදා ගුවනෙහි සැරිසැරීමට නොහැකි වීමය. තම හිසට ගව් ගණනක් ඉහළ අහසේ පියාසර කරනා සියොතුන් දුටු ආදි මානවයා, අනාගත මානවයා අහස් කුස තුළ තම අණසක පතුරවනු ඇතැයි කිසිදා නොසිතන්නට ඇත. අහසෙහි පියාසර කරන සියොතුන් සොබාදම් මාතාවගේ විශ්මයජනක නිමැවුමක් ලෙස සලකා ඔවුන් දේවත්වයෙන් ඇදහූ ඉතිහාසයක් ද අපට උරුමය. නිදහසේ නභෝගැබ සැරිසරන සියොතුන් දැක කොතෙකුත් දෙනා වලාකුළු අතර සැරිසරන්නට සිහින දකින්නට ඇද්ද? දඬුමොණරයෙන් අහස ජයගත් රාවණා රජුගේ කතා පුරාවෘත්තයන්හි අප අසා ඇත. සීගිරි පර්වතය මුදුනේ සිට රූමස්සල කඳු වැටිය දක්වා අහසින් ගිය පුවත් සැබෑවක් ද යන්න පසෙක තැබුව ද, එම සංකල්පය තුළ ගැබ්ව ඇත්තේ අහස ජය ගැනීමට අපේ මුතුන් මිත්තන් තුළ වූ පරිකල්පනය නොවේ ද? ගුවන තරණය කිරීමේ සිහිනය යථාර්ථයක් කරනු වස් පළමු ප්‍රයත්නය දරන ලද්දේ රයිට් සොහොයුරන් විසිනි. ජෛව ඉංජිනේරු විද්‍යාත්මකව නිර්මාණය වූ කුරුල්ලකුගේ ආකෘතියක් අනුසාරයෙන් නිමවූ ගුවන් යානය, අද වන විට එක්වර මගීන් දහසක් පමණ රැගෙන යන මට්ටමට තාක්ෂණික දියුණුවක් විදහාපායි."""
    }
]


def aggregate_metrics(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate comparison metrics across all test texts."""
    
    metrics = {
        "lime": {
            "total_time": 0.0,
            "avg_confidence": 0.0,
            "success_count": 0,
            "important_token_counts": [],
        },
        "shap": {
            "total_time": 0.0,
            "avg_confidence": 0.0,
            "success_count": 0,
            "important_token_counts": [],
        },
        "agreement": {
            "prediction_agreements": 0,
            "token_pearson_correlations": [],
            "token_spearman_correlations": [],
            "sign_agreements": [],
            "top_k_jaccard_5": [],
            "top_k_jaccard_10": [],
            "top_k_jaccard_15": [],
        }
    }
    
    for report in reports:
        lime = report["lime"]
        shap = report["shap"]
        comp = report["comparison"]
        
        # LIME metrics
        metrics["lime"]["total_time"] += comp["lime_elapsed_seconds"]
        metrics["lime"]["avg_confidence"] += lime["confidence"]
        if lime.get("success", False):
            metrics["lime"]["success_count"] += 1
        metrics["lime"]["important_token_counts"].append(comp["lime_important_token_count"])
        
        # SHAP metrics
        metrics["shap"]["total_time"] += comp["shap_elapsed_seconds"]
        metrics["shap"]["avg_confidence"] += shap["confidence"]
        if shap.get("success", False):
            metrics["shap"]["success_count"] += 1
        metrics["shap"]["important_token_counts"].append(comp["shap_important_token_count"])
        
        # Agreement metrics
        if comp["prediction_agreement"]:
            metrics["agreement"]["prediction_agreements"] += 1
        
        if comp["token_pearson_correlation"] is not None:
            metrics["agreement"]["token_pearson_correlations"].append(
                comp["token_pearson_correlation"]
            )
        if comp["token_spearman_correlation"] is not None:
            metrics["agreement"]["token_spearman_correlations"].append(
                comp["token_spearman_correlation"]
            )
        if comp["token_sign_agreement"] is not None:
            metrics["agreement"]["sign_agreements"].append(comp["token_sign_agreement"])
        
        if "top_5" in comp["top_k_jaccard"]:
            metrics["agreement"]["top_k_jaccard_5"].append(comp["top_k_jaccard"]["top_5"])
        if "top_10" in comp["top_k_jaccard"]:
            metrics["agreement"]["top_k_jaccard_10"].append(comp["top_k_jaccard"]["top_10"])
        if "top_15" in comp["top_k_jaccard"]:
            metrics["agreement"]["top_k_jaccard_15"].append(comp["top_k_jaccard"]["top_15"])
    
    # Calculate averages
    n = len(reports)
    metrics["lime"]["avg_confidence"] /= n
    metrics["shap"]["avg_confidence"] /= n
    
    # Calculate agreement statistics
    if metrics["agreement"]["token_pearson_correlations"]:
        metrics["agreement"]["avg_token_pearson"] = np.mean(
            metrics["agreement"]["token_pearson_correlations"]
        )
        metrics["agreement"]["std_token_pearson"] = np.std(
            metrics["agreement"]["token_pearson_correlations"]
        )
    
    if metrics["agreement"]["token_spearman_correlations"]:
        metrics["agreement"]["avg_token_spearman"] = np.mean(
            metrics["agreement"]["token_spearman_correlations"]
        )
        metrics["agreement"]["std_token_spearman"] = np.std(
            metrics["agreement"]["token_spearman_correlations"]
        )
    
    if metrics["agreement"]["sign_agreements"]:
        metrics["agreement"]["avg_sign_agreement"] = np.mean(
            metrics["agreement"]["sign_agreements"]
        )
    
    if metrics["agreement"]["top_k_jaccard_5"]:
        metrics["agreement"]["avg_jaccard_5"] = np.mean(
            metrics["agreement"]["top_k_jaccard_5"]
        )
    if metrics["agreement"]["top_k_jaccard_10"]:
        metrics["agreement"]["avg_jaccard_10"] = np.mean(
            metrics["agreement"]["top_k_jaccard_10"]
        )
    if metrics["agreement"]["top_k_jaccard_15"]:
        metrics["agreement"]["avg_jaccard_15"] = np.mean(
            metrics["agreement"]["top_k_jaccard_15"]
        )
    
    return metrics


def print_summary_report(metrics: Dict[str, Any], text_count: int) -> None:
    """Print a formatted summary report of all comparisons."""
    
    print("\n" + "=" * 80)
    print("XAI PERFORMANCE TEST SUMMARY")
    print("=" * 80)
    print(f"\nTest Texts: {text_count}")
    
    print("\n--- LIME PERFORMANCE ---")
    print(f"Total Time: {metrics['lime']['total_time']:.3f}s")
    print(f"Average Time per Text: {metrics['lime']['total_time'] / text_count:.3f}s")
    print(f"Average Confidence: {metrics['lime']['avg_confidence']:.4f}")
    print(f"Success Rate: {metrics['lime']['success_count']}/{text_count}")
    print(f"Avg Important Tokens: {np.mean(metrics['lime']['important_token_counts']):.1f}")
    
    print("\n--- SHAP PERFORMANCE ---")
    print(f"Total Time: {metrics['shap']['total_time']:.3f}s")
    print(f"Average Time per Text: {metrics['shap']['total_time'] / text_count:.3f}s")
    print(f"Average Confidence: {metrics['shap']['avg_confidence']:.4f}")
    print(f"Success Rate: {metrics['shap']['success_count']}/{text_count}")
    print(f"Avg Important Tokens: {np.mean(metrics['shap']['important_token_counts']):.1f}")
    
    print("\n--- AGREEMENT METRICS ---")
    print(f"Prediction Agreement: {metrics['agreement']['prediction_agreements']}/{text_count}")
    
    if "avg_token_pearson" in metrics["agreement"]:
        print(f"Avg Token Pearson Correlation: {metrics['agreement']['avg_token_pearson']:.4f} "
              f"(±{metrics['agreement']['std_token_pearson']:.4f})")
    
    if "avg_token_spearman" in metrics["agreement"]:
        print(f"Avg Token Spearman Correlation: {metrics['agreement']['avg_token_spearman']:.4f} "
              f"(±{metrics['agreement']['std_token_spearman']:.4f})")
    
    if "avg_sign_agreement" in metrics["agreement"]:
        print(f"Avg Token Sign Agreement: {metrics['agreement']['avg_sign_agreement']:.2%}")
    
    if "avg_jaccard_5" in metrics["agreement"]:
        print(f"Avg Jaccard Similarity (top-5): {metrics['agreement']['avg_jaccard_5']:.4f}")
    
    if "avg_jaccard_10" in metrics["agreement"]:
        print(f"Avg Jaccard Similarity (top-10): {metrics['agreement']['avg_jaccard_10']:.4f}")
    
    if "avg_jaccard_15" in metrics["agreement"]:
        print(f"Avg Jaccard Similarity (top-15): {metrics['agreement']['avg_jaccard_15']:.4f}")
    
    print("\n--- EFFICIENCY COMPARISON ---")
    lime_total = metrics["lime"]["total_time"]
    shap_total = metrics["shap"]["total_time"]
    if lime_total > 0 and shap_total > 0:
        if lime_total < shap_total:
            speedup = shap_total / lime_total
            print(f"LIME is {speedup:.2f}x faster than SHAP")
        else:
            speedup = lime_total / shap_total
            print(f"SHAP is {speedup:.2f}x faster than LIME")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    
    # Determine which technique is better based on multiple criteria
    lime_score = 0
    shap_score = 0
    
    # Speed (LIME is generally faster)
    if lime_total < shap_total:
        lime_score += 2
    else:
        shap_score += 2
    
    # Prediction agreement
    if metrics["agreement"]["prediction_agreements"] > text_count / 2:
        lime_score += 1
        shap_score += 1
    
    # Correlation (higher is better for consistency)
    if "avg_token_pearson" in metrics["agreement"]:
        if metrics["agreement"]["avg_token_pearson"] > 0.5:
            lime_score += 1
            shap_score += 1
    
    # Success rate
    if metrics["lime"]["success_count"] > metrics["shap"]["success_count"]:
        lime_score += 1
    else:
        shap_score += 1
    
    if lime_score > shap_score:
        print(f"\nBased on the test results, LIME appears to perform better overall.")
        print(f"  - Faster execution")
        print(f"  - Lower computational overhead")
        print(f"  - Good agreement with SHAP on important tokens")
    elif shap_score > lime_score:
        print(f"\nBased on the test results, SHAP appears to provide better explanations.")
        print(f"  - More theoretically grounded explanations")
        print(f"  - Better agreement metrics")
    else:
        print(f"\nBoth methods show comparable performance characteristics.")
        print(f"  - Choose based on your specific requirements:")
        print(f"    * LIME: Faster, lower memory footprint")
        print(f"    * SHAP: More principled approach, better for production")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test and compare LIME vs SHAP performance on multiple Sinhala texts."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of perturbation samples for explainers.",
    )
    parser.add_argument(
        "--num-features",
        type=int,
        default=10,
        help="Number of features to request.",
    )
    parser.add_argument(
        "--output",
        help="Path to save detailed results as JSON.",
    )
    args = parser.parse_args()
    
    reports = []
    
    print("\n" + "=" * 80)
    print("TESTING XAI TECHNIQUES ON SINHALA TEXTS")
    print("=" * 80)
    
    for i, text_config in enumerate(TEST_TEXTS, 1):
        print(f"\n[{i}/{len(TEST_TEXTS)}] Testing: {text_config['topic']}")
        print(f"Text: {text_config['name']}")
        print("-" * 80)
        
        try:
            report = run_comparison(
                text_config["text"],
                args.num_samples,
                args.num_features,
                top_k=5,
            )
            reports.append(report)
            
            comp = report["comparison"]
            print(f"✓ Predictions Agree: {comp['prediction_agreement']}")
            print(f"  LIME time: {comp['lime_elapsed_seconds']:.3f}s")
            print(f"  SHAP time: {comp['shap_elapsed_seconds']:.3f}s")
            print(f"  Speedup: {comp['speedup_ratio']:.2f}x" if comp['speedup_ratio'] else "  Speedup: N/A")
            if comp["token_pearson_correlation"] is not None:
                print(f"  Token Pearson: {comp['token_pearson_correlation']:.4f}")
        
        except Exception as e:
            logger.error(f"Failed to test {text_config['name']}: {e}")
            continue
    
    if reports:
        metrics = aggregate_metrics(reports)
        print_summary_report(metrics, len(reports))
        
        if args.output:
            output_data = {
                "test_texts": len(reports),
                "aggregate_metrics": {
                    "lime": {
                        "total_time": float(metrics["lime"]["total_time"]),
                        "avg_confidence": float(metrics["lime"]["avg_confidence"]),
                        "success_count": int(metrics["lime"]["success_count"]),
                    },
                    "shap": {
                        "total_time": float(metrics["shap"]["total_time"]),
                        "avg_confidence": float(metrics["shap"]["avg_confidence"]),
                        "success_count": int(metrics["shap"]["success_count"]),
                    },
                    "agreement": {
                        "prediction_agreements": int(metrics["agreement"]["prediction_agreements"]),
                        "avg_token_pearson": float(metrics["agreement"].get("avg_token_pearson", 0)),
                        "avg_token_spearman": float(metrics["agreement"].get("avg_token_spearman", 0)),
                        "avg_sign_agreement": float(metrics["agreement"].get("avg_sign_agreement", 0)),
                    }
                },
                "detailed_reports": reports,
            }
            
            output_path = Path(args.output)
            output_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n✓ Detailed results saved to {output_path}")
    else:
        print("\nNo successful comparisons were generated.")


if __name__ == "__main__":
    main()
