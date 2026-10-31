"""Transcribe the visual review decisions of 2026-10-14 (no detector labels).

Background proposals were geometrically transferred from two manually drawn
stationary motorcycles in frame 585, then individually checked in five sheets.
Only the explicit indices below have passed that visual check.
"""
import json

from prepare_dataset import ROOT, write_json


def main():
    proposals = json.loads((ROOT/'review_round3/background_proposals.json').read_text())
    approved_background = [585,600,615,630,645,660,675,690,705,720,750,765,780,795,
        861,876,891,906,921,936,951,966,996,1011,1026,1041,1056,1071,1086,1116,
        1614,1621,1630,1644,1654,1660,653]
    original = []
    for i in approved_background:
        entry = dict(index=i, status='corrected', visual_reviewed=True,
            append=proposals[str(i)]['append'],
            evidence='review_round3/sheet_00..23.jpg + background_00..04.jpg; static background boxes individually checked',
            reason='Rà toàn ảnh: người chính đang ngồi xe, mũ/tóc rõ; bổ sung xe đỏ và xe trắng đỗ trong nền khi nhìn thấy. Không gán rider cho xe đỗ.')
        if i in [951,966,1041,653]:
            entry['drop'] = {951:[4],966:[2],1041:[2],653:[1,3]}[i]
            entry['reason'] += (' Xóa box rider giả trên xe đỗ hoặc box trùng; xe đỗ có box lớp 0 riêng.')
        original.append(entry)
    for i in [1101,1416,1425,1434,1455,1464,1482,1491,1509,1698,
              1218,1231,1248,1260,1281,1293,1318,1347,1374,1386,1395,1401,1676,1783,
              9,12,1564,1707]:
        original.append(dict(index=i, status='accepted', visual_reviewed=True,
            reason='Rà toàn ảnh và box: đối tượng chính rõ, không có rider bị bỏ sót hoặc người đi bộ bị gán rider. Giữ phân lớp theo mũ đang đội.'))
    # Include the visible mirrors in the close rear views rather than labeling only the wheel/seat.
    bike_replacements = {
        585:[0,75,337,239,699], 1101:[0,741,160,1040,677],
        1425:[0,600,245,876,700], 1455:[0,165,292,695,716],
        1464:[0,655,112,820,538], 1482:[0,716,311,1041,716],
        1491:[0,737,229,1016,705], 1509:[0,573,245,922,720],
        1698:[0,607,91,824,516], 1293:[0,873,365,1028,570],
    }
    for entry in original:
        if entry['index'] in bike_replacements:
            entry['replace'] = {'0':bike_replacements[entry['index']]}
            entry['reason'] += ' Mở box xe tới phần gương/thân xe nhìn thấy.'
    original.extend([
        dict(index=4,status='corrected',visual_reviewed=True,reclass={'1':1},
             reason='Hai người ngồi trên xe đều đội mũ đen; sửa lớp 2 thành lớp 1 và giữ hai box riêng.'),
        dict(index=6,status='corrected',visual_reviewed=True,replace={'1':[1,464,65,797,578]},
             reason='Rider đội mũ; box cũ chỉ có đầu và thân trên, mở đến bàn chân nhìn thấy.'),
        dict(index=1683,status='corrected',visual_reviewed=True,append=[[0,1143,337,1234,524]],
             reason='Giữ sửa lượt 1: xóa hai box người đứng quầy. Bổ sung xe phải còn thiếu, mỗi người ngồi xe một box; cả ba đều đội mũ.'),
        dict(index=572,status='pending',
             reason='Hai rider rõ nhưng xe đỏ đỗ trong nền bị tay/người phía trước che khuất; box nháp chồng sang người. Tách cả ảnh, dùng frame 575 đã rà đầy đủ.'),
        dict(index=575,status='corrected',visual_reviewed=True,
             append=[[0,313,114,366,177],[0,340,118,417,181],[0,480,139,549,200]],
             reason='Hai người tóc trần, tách box từng người; bổ sung ba xe máy đỗ trong nền, loại xe đạp khỏi lớp xe máy.'),
    ])
    pending_reasons = {
        3:'Box gộp hai người; dùng biến thể 4 đã sửa, giữ bản này chờ duyệt để tránh nhân bản cùng ảnh.',
        1358:'Đầu rider bên phải giống mũ lưỡi trai; chưa đủ bằng chứng phân biệt với mũ bảo hiểm. Tách cả ảnh.',
        1705:'Vùng đầu/thân phía trước rider bị che; chưa chắc một hay hai người. Tách cả ảnh.',
        1565:'Nhiều xe đỗ nhỏ trong nền chưa được gán đủ; box hai rider chưa chuẩn toàn thân.',
        21:'Rider ở xa không đủ rõ để phân lớp mũ; giữ cả ảnh ngoài các tập hoạt động.',
        48:'Box có thể gộp người lái/ngồi sau; cần kiểm tra vùng đầu bị che.',
        51:'Rider sát camera chỉ còn tay và gương trong khung, không rõ mũ; tách cả ảnh.',
        735:'Rider sát mép ảnh, vùng đầu bị cắt; không suy lớp từ frame khác.',
        981:'Chỉ thấy một phần người sát camera, xe ngoài khung; không đủ để gán rider/xe nhất quán.',
        1500:'Đầu rider nằm ngoài ảnh; không suy lớp theo video lân cận.',
    }
    selected = json.loads((ROOT/'review_round3/selected_indices.json').read_text())
    active = {e['index'] for e in original}
    for i in selected:
        if i not in active:
            original.append(dict(index=i,status='pending',reason=pending_reasons.get(i,
                'Chưa gán đủ/khẳng định được các xe hoặc người nhỏ trong nền; giữ cả ảnh ngoài tập hoạt động.')))
    previous = json.loads((ROOT/'annotations_round2.json').read_text(encoding='utf-8'))
    external = []
    def add(i, boxes, canvas, evidence, reason, split='train'):
        external.append(dict(index=i, boxes=boxes, canvas=canvas, country='VN',
            location_evidence=evidence, visual_reviewed=True, reason=reason, split=split))
    for i in [3,4,14]:
        old = next(e for e in previous['external'] if e['index']==i)
        evidence = {3:'Commons category Motorcycles in Vietnam; source revision archived in vietnam_provenance.json.',
            4:'Cùng tác giả/chuyến đi và xe biển 49-E1 552.27 như ảnh 3; biển số Việt Nam và cờ Việt Nam. Chỉ đưa train, gom cả tác giả.',
            14:'Mô tả nguồn ghi rõ motorcycle on the street at Vietnam.'}[i]
        add(i,old['boxes'],[1280,960] if i==14 else [1280,853],evidence,old['reason'])
    add(48, [[0,378,241,1089,714],[2,582,87,964,635],
             [0,0,188,168,464],[0,25,231,291,525],[0,286,242,564,551]], [1280,805],
        'Mô tả và tọa độ nguồn: Dương Đông, Phú Quốc, Việt Nam; Commons revision archived.',
        'Một người ngồi xe đội mũ vải chống nắng, không phải mũ bảo hiểm. Rà crop nền: ba xe máy đỗ, không có rider nền.', 'test')
    add(66, [[0,314,110,542,410],[0,522,72,771,455],[0,867,134,1087,525],
             [0,1004,125,1192,384],[0,1106,156,1280,486]], [1280,960],
        'Mô tả: Quang Ngai, Vjetnamio; category Vehicles in Quang Ngai City.',
        'Năm xe máy đỗ. Trẻ em ngồi dưới đất/đứng cửa không thuộc rider; xe đạp sau xe Cub không gán lớp xe máy.')
    add(76, [[0,0,0,1280,863]], [1280,868], 'Tên và mô tả nguồn: Minsk motorcycle in Vietnam.',
        'Một xe máy đỗ, không có người ngồi xe; găng tay trên yên không phải rider.')
    add(77, [[0,70,208,916,943]], [939,960], 'Mô tả nguồn: Halong City ferry boat, Vietnam.',
        'Một xe máy chở hàng trên phà, không có rider; không gán hàng hóa thành người.')
    add(85, [[0,603,379,1017,645],[1,778,299,905,658],[1,870,298,983,577]], [1280,720],
        'Tên và category nguồn: Motorcycle in Vietnam; cùng tác giả Vinnie Cartabiano với ảnh 3/4.',
        'Hai người đang ngồi cùng xe, cả hai đội mũ. Người chống chân vẫn đang ngồi trên yên.')
    add(136, [[0,40,275,1019,951],[1,333,22,803,866]], [1019,960],
        'Commons categories Motorcycles in Vietnam / Freight motorcycles in Vietnam; revision archived.',
        'Một rider đội mũ xanh, một xe; lốp xe chở phía sau không phải người.')
    add(143, [[0,119,190,838,928],[1,136,0,536,783]], [873,960],
        'Commons categories Motorcycles in Vietnam / Freight motorcycles in Vietnam; revision archived.',
        'Một rider đội mũ, một xe chở gốm; không thêm rider từ hàng hóa.')
    add(145, [[0,246,281,858,889],[1,268,110,629,756]], [960,960],
        'Tên, mô tả và category nguồn: Vietnamese Rooster Transporter / Motorcycles in Vietnam.',
        'Một người đội mũ kín đầu đang ngồi xe; các lồng hàng không phải người.')
    add(163, [[0,119,198,1094,840]], [1280,853],
        'Commons categories Graffiti in Vietnam / Motorcycles in Vietnam; biển số Việt Nam; revision archived.',
        'Một xe Honda đỗ, không có người ngồi; dùng mẫu chống báo rider giả.')
    write_json(ROOT/'annotations_v2.json', dict(review_date='2026-10-14',
        policy='Rà toàn ảnh; mỗi người ngồi xe một box; mũ vải/lưỡi trai không phải mũ bảo hiểm. Ảnh mơ hồ tách nguyên ảnh.',
        original=original, external=external))
    print('Active originals:',sum(e['status']!='pending' for e in original),'VN external:',len(external))


if __name__ == '__main__':
    main()
