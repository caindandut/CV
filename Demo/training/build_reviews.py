"""Materialize explicit visual-review decisions, never model-generated labels."""
import json
import re
from pathlib import Path
from prepare_dataset import ROOT, write_json, group_rows


def camera_family(row):
    name=Path(row['image']).name
    if re.match(r'Image\d+_',name,re.I):
        return 'scene:public_road'
    if re.match(r'image_\d+_',name,re.I) or re.match(r'IMG_503[0-7]',name,re.I):
        return 'scene:campus_forecourt'
    if re.match(r'IMG_50(?:70|73|75|76)',name,re.I) or re.match(r'IMG_52(?:7[5-9]|8[0-9]|9[01])',name,re.I):
        return 'scene:campus_parking'
    if re.match(r'IMG_51(?:2[6-9]|3\d|4[0-3])',name,re.I):
        return 'scene:campus_gate'
    return None


def main():
    rows=json.loads((ROOT/'inventory.json').read_text(encoding='utf-8'))
    decisions=json.loads((ROOT/'curation_decisions.json').read_text(encoding='utf-8'))
    accepted={i for a,b in decisions['accepted_ranges'] for i in range(a,b+1)}
    accepted.difference_update(decisions['excluded_after_first_pass'])
    reviews={}
    for i,row in enumerate(rows):
        family=camera_family(row)
        row['original_source']=row.get('original_source',row['source'])
        if family:
            row['source']=family
        reason='Cần rà bằng mắt chi tiết; chưa xác nhận nhãn.'
        for item in decisions['pending_ranges']:
            if item['range'][0]<=i<=item['range'][1]:
                reason=item['reason']
        if i in accepted:
            reason='Rà contact sheet: người/xe và trạng thái mũ nhìn rõ; chưa thấy lỗi cần sửa trong lượt rà này.'
        reviews[row['id']]=dict(status='accepted' if i in accepted else 'pending',reason=reason,
                                label_sha256=row['label_sha256'],source_verified=family is not None,
                                evidence=f'review_sheets/sheet_{i//24:03d}.jpg',
                                review_level='contact_sheet_first_pass',inventory_index=i)
    def change(index, labels, reason, pending=False):
        reviews[rows[index]['id']].update(status='pending' if pending else 'corrected',labels=labels,
                                         reason=reason,review_level='full_image_visual',
                                         evidence=f'review_sheets/detail_{index}.jpg')
    # Explicitly viewed full-size, class-only corrections preserve geometry.
    boxes=[list(b) for b in rows[3]['labels']]
    for index,b in enumerate(boxes):
        if b[0]==2:
            # Explicit full-image bounds: include visible helmet, back, arms and feet.
            x1,y1,x2,y2=275,4,790,593
            w,h=rows[3]['width'],rows[3]['height']
            boxes[index]=[1,(x1+x2)/2/w,(y1+y2)/2/h,(x2-x1)/w,(y2-y1)/h]
    change(3,boxes,'Ảnh lớn xác nhận mũ đen: sửa lớp 2 thành 1, mở box người để bao đủ phần đầu/lưng/tay/chân nhìn thấy thay vì chỉ nửa phải người.')
    # Full-size inspection confirms a black helmet in 795/796/797: preserve class 1.
    for i in [795,796,797]:
        reviews[rows[i]['id']].update(status='accepted',review_level='full_image_visual',
                                     evidence=f'review_sheets/detail_{i}.jpg',
                                     reason='Ảnh lớn xác nhận mũ đen trên đầu; giữ nguyên lớp 1, không suy từ thumbnail.')
    for i in [1386,1387,1388]:
        boxes=[list(b) for b in rows[i]['labels']]
        for b in boxes:
            if b[0]==2:
                b[0]=1
        change(i,boxes,'Người áo xanh bên trái đội mũ trắng sọc xanh; sửa lớp 2 thành 1.',i==1387)
    # Correct clear mistakes but retain whole image pending when annotation remains incomplete.
    boxes=[list(b) for b in rows[1598]['labels'] if b[0]!=2]
    for b in boxes:
        if b[0]==1:
            b[0]=2
    change(1598,boxes,'Bỏ hai box người đứng cạnh xe/người đi bộ; người lái không đội mũ. Còn thiếu xe đỗ nền nên giữ cả ảnh chờ gán đủ.',True)
    boxes=[list(b) for b in rows[1683]['labels'] if b[0]!=2]
    change(1683,boxes,'Bỏ hai box cùng người phụ nữ đứng quầy. Còn thiếu box xe bên phải nên giữ cả ảnh chờ gán đủ.',True)
    # Any accepted gate frame with fewer bikes than riders needs a detailed missing-box review.
    for row in rows:
        review=reviews[row['id']]
        labels=review.get('labels',row['labels'])
        bikes=sum(b[0]==0 for b in labels)
        riders=len(labels)-bikes
        if review['status']!='pending' and (bikes==0 and riders or row['source']=='scene:campus_gate' and bikes<riders):
            review.update(status='pending',reason=review['reason']+' Cần rà thêm box xe thiếu; tách cả ảnh.')
        if review['status']!='pending' and row['source']=='scene:campus_forecourt' and any(b[0]==2 for b in labels):
            review.update(status='pending',reason='Rà chi tiết: box lớp 2 rất nhỏ ở lối đi/nền có khả năng là pedestrian, chưa chứng minh đang ngồi trên xe; tách cả ảnh.')
    raw_rows=[dict(row,source=row['original_source']) for row in rows]
    links=group_rows(raw_rows)
    links+=group_rows(rows)
    write_json(ROOT/'inventory.json',rows)
    write_json(ROOT/'near_duplicate_links.json',links)
    write_json(ROOT/'reviews.json',reviews)
    print('reviewed first pass:',len(reviews),'accepted/corrected:',sum(r['status']!='pending' for r in reviews.values()))


if __name__=='__main__':
    main()
