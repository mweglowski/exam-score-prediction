import React, { useState } from 'react';

const THEME = {
  bg: '#F5F5F5',
  card: '#FFFFFF',
  text: '#333333',
  subText: '#555555',
  accent: '#4C72B0',
  border: '#EAEAF2',
  success: '#55A868',
  error: '#C44E52',
  hover: '#3b5a8c'
};

const OPTIONS = {
  gender: ['male', 'female', 'other'],
  study_method: ['online videos', 'self-study', 'coaching', 'group study', 'mixed'],
  course: ['b.sc', 'diploma', 'bca', 'b.com', 'ba', 'bba', 'b.tech'],
  facility_rating: ['low', 'medium', 'high'],
  sleep_quality: ['poor', 'average', 'good'],
  exam_difficulty: ['easy', 'moderate', 'hard'],
  internet_access: ['no', 'yes']
};

const INITIAL_STATE = {
  age: 20,
  study_hours: 5.0,
  class_attendance: 80,
  sleep_hours: 7.0,
  study_method: 'self-study',
  course: 'b.sc',
  gender: 'male',
  facility_rating: 'medium',
  sleep_quality: 'average',
  exam_difficulty: 'moderate',
  internet_access: 'yes'
};

function App() {
  const [formData, setFormData] = useState(INITIAL_STATE);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setPrediction(null);

    try {
      const orderedKeys = [
        'age', 'study_hours', 'class_attendance', 'sleep_hours',
        'study_method', 'course', 'gender',
        'facility_rating', 'sleep_quality', 'exam_difficulty', 'internet_access'
      ];

      const features = orderedKeys.map(key => {
        const val = formData[key];
        return isNaN(Number(val)) ? val : parseFloat(val);
      });

      console.log("Sending features:", features);

      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: features }),
      });

      const data = await response.json();

      if (response.ok) {
        setPrediction(data.prediction);
      } else {
        setError(data.error || 'Something went wrong');
      }
    } catch (err) {
      setError('Failed to connect to server');
    }
  };


  const SectionTitle = ({ children }) => (
    <h3 style={{ 
      color: THEME.accent, 
      borderBottom: `2px solid ${THEME.border}`, 
      paddingBottom: '10px', 
      marginBottom: '20px',
      fontSize: '1.1rem',
      fontWeight: '600',
      letterSpacing: '0.5px'
    }}>
      {children}
    </h3>
  );

  const renderNumber = (label, name) => (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      <label style={{ marginBottom: '6px', fontSize: '0.9rem', color: THEME.subText, fontWeight: '500' }}>
        {label}
      </label>
      <input 
        type="number" 
        name={name} 
        value={formData[name]} 
        onChange={handleChange}
        step="0.1"
        style={{ 
          padding: '10px', 
          borderRadius: '6px', 
          border: `1px solid ${THEME.border}`, 
          backgroundColor: '#FAFAFA',
          fontFamily: 'Poppins, sans-serif',
          fontSize: '0.95rem',
          outline: 'none',
          transition: 'border-color 0.2s'
        }}
      />
    </div>
  );

  const renderSelect = (label, name, options) => (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      <label style={{ marginBottom: '6px', fontSize: '0.9rem', color: THEME.subText, fontWeight: '500' }}>
        {label}
      </label>
      <select 
        name={name} 
        value={formData[name]} 
        onChange={handleChange}
        style={{ 
          padding: '10px', 
          borderRadius: '6px', 
          border: `1px solid ${THEME.border}`, 
          backgroundColor: '#FAFAFA',
          fontFamily: 'Poppins, sans-serif',
          fontSize: '0.95rem',
          outline: 'none',
          cursor: 'pointer'
        }}
      >
        {options.map(opt => (
          <option key={opt} value={opt}>{opt}</option>
        ))}
      </select>
    </div>
  );

  return (
    <>
      <style>
        {`@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');`}
      </style>

      <div style={{ 
        minHeight: '100vh', 
        backgroundColor: THEME.bg, 
        fontFamily: 'Poppins, sans-serif',
        padding: '40px 20px',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'flex-start'
      }}>
        
        <div style={{ 
          width: '100%', 
          maxWidth: '700px', 
          backgroundColor: THEME.card, 
          borderRadius: '12px', 
          boxShadow: '0 10px 30px rgba(0,0,0,0.05)',
          padding: '40px'
        }}>
          
          <h1 style={{ 
            textAlign: 'center', 
            color: THEME.text, 
            marginBottom: '40px', 
            fontSize: '2rem',
            fontWeight: '700' 
          }}>
            Exam Score Predictor
          </h1>
          
          <form onSubmit={handleSubmit}>
            
            <div style={{ marginBottom: '40px' }}>
              <SectionTitle>Numeric Variables</SectionTitle>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                {renderNumber("Age", "age")}
                {renderNumber("Study Hours", "study_hours")}
                {renderNumber("Attendance (%)", "class_attendance")}
                {renderNumber("Sleep Hours", "sleep_hours")}
              </div>
            </div>

            <div style={{ marginBottom: '40px' }}>
              <SectionTitle>Categorical Variables</SectionTitle>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                {renderSelect("Gender", "gender", OPTIONS.gender)}
                {renderSelect("Study Method", "study_method", OPTIONS.study_method)}
                {renderSelect("Course", "course", OPTIONS.course)}
                {renderSelect("Internet Access", "internet_access", OPTIONS.internet_access)}
                {renderSelect("Facility Rating", "facility_rating", OPTIONS.facility_rating)}
                {renderSelect("Sleep Quality", "sleep_quality", OPTIONS.sleep_quality)}
                {renderSelect("Exam Difficulty", "exam_difficulty", OPTIONS.exam_difficulty)}
              </div>
            </div>

            <button 
              type="submit" 
              style={{ 
                width: '100%', 
                padding: '16px', 
                backgroundColor: THEME.accent, 
                color: 'white', 
                fontSize: '1rem',
                fontWeight: '600',
                border: 'none', 
                borderRadius: '8px', 
                cursor: 'pointer', 
                boxShadow: '0 4px 6px rgba(76, 114, 176, 0.2)',
                transition: 'transform 0.1s ease',
                fontFamily: 'Poppins, sans-serif'
              }}
              onMouseOver={(e) => e.target.style.backgroundColor = THEME.hover}
              onMouseOut={(e) => e.target.style.backgroundColor = THEME.accent}
            >
              Predict Score
            </button>
          </form>

          {prediction !== null && (
            <div style={{ 
              marginTop: '30px', 
              padding: '20px', 
              backgroundColor: '#EDF7F0', 
              border: `1px solid ${THEME.success}`,
              color: THEME.success, 
              borderRadius: '8px', 
              textAlign: 'center' 
            }}>
              <span style={{ fontSize: '0.9rem', fontWeight: '500', display: 'block', marginBottom: '5px' }}>PREDICTED SCORE</span>
              <span style={{ fontSize: '2.5rem', fontWeight: '700' }}>{Math.round(prediction * 100) / 100}</span>
            </div>
          )}

          {error && (
            <div style={{ 
              marginTop: '30px', 
              padding: '15px', 
              backgroundColor: '#FDEDED', 
              border: `1px solid ${THEME.error}`,
              color: THEME.error, 
              borderRadius: '8px', 
              textAlign: 'center',
              fontWeight: '500'
            }}>
              Error: {error}
            </div>
          )}
        </div>
      </div>
    </>
  );
}

export default App;