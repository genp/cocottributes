### Genevieve's Implementation

`ann_vecs` is dict of attribute vectors. The key is used to index into the `patch_id_to_ann_id` dict to get the ann id from the MS COCO annotation dataset (the original MS COCO dataset).

    img_attrs = attr_data['ann_vecs'][idx]
    ann_id = attr_data['patch_id_to_ann_id'][idx]
    
`split` is a dict which tells us whether a the image for the attribute vector is in the train split or val split.
 
Given the ann_id above, we now have to search for the same ann_id (given by id in the annotation dataset):

    anns = [a for a in annotations['annotations'] if a['id'] == ann_id]
    # anns should have only one entry since id is unique
    ann = anns[0]
    # Get the image name
    img_id = ann['image_id']
    
 To map the attributes in `ann_vecs` to the actual attributes, we need to sort the attributes list in `attributes`.

### Varun's Implementation

In the attributes dataset, `ann_attrs` is a dict where each key is the id (ann_id) of the annotation/object from the annotation dataset. The value for the ann_id will be a dictionary which has the following fields:

- `attrs_vector` which is the list of attribute values from the ELA algorithm.
- `split` which tells us which split does the image belongs to.

The `attributes` list will be saved in the sorted order, sorted on id. This is the same order in which the `attrs_vector` is ordered.

Thus to get the attribute vector for any annotation would be:

    ann = annotations['annotation'][idx]  # idx is a valid index in the annotations dataset
    attr = attributes['ann_attrs'][ann['id']]
    attr_vector = attr['attrs_vector']
    split = attr['split']  # this is partially redundant since our idx is obtained from either the train or val split
