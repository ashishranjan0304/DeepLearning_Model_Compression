import React, { useState } from 'react';

const CommandForm = () => {
  const [formData, setFormData] = useState({
    nproc_per_node: 3,
    batch_size: 2,
    no_aux_loss: false,
    eval: true,
    resume_path: '/home/ashishr/DETR-Object-Detection/detr/RN1_RD0_LB10_LE140_LI1/best_checkpoint.pth',
    coco_path: '../open-images-bus-trucks/'
  });

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData({
      ...formData,
      [name]: type === 'checkbox' ? checked : value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const response = await fetch('/run-command', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(formData)
    });
    const result = await response.json();
    alert(result.status);
  };

  return (
    <form onSubmit={handleSubmit}>
      <label>
        Number of Processes per Node:
        <input type="number" name="nproc_per_node" value={formData.nproc_per_node} onChange={handleChange} />
      </label>
      <br />
      <label>
        Batch Size:
        <input type="number" name="batch_size" value={formData.batch_size} onChange={handleChange} />
      </label>
      <br />
      <label>
        <input type="checkbox" name="no_aux_loss" checked={formData.no_aux_loss} onChange={handleChange} />
        No Aux Loss
      </label>
      <br />
      <label>
        <input type="checkbox" name="eval" checked={formData.eval} onChange={handleChange} />
        Evaluate
      </label>
      <br />
      <label>
        Resume Path:
        <input type="text" name="resume_path" value={formData.resume_path} onChange={handleChange} />
      </label>
      <br />
      <label>
        COCO Path:
        <input type="text" name="coco_path" value={formData.coco_path} onChange={handleChange} />
      </label>
      <br />
      <button type="submit">Run Command</button>
    </form>
  );
};

export default CommandForm;
