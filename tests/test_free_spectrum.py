import math
import pytest
from lattice_scattering.kinematics import LatticeFrame
from lattice_scattering.kinematics.free import free_two_body_states


def test_rest_shells_and_labelled_degeneracy():
    result=free_two_body_states(mass1_at=.45,mass2_at=.55,frame=LatticeFrame(16,1.),
                                energy_max_at=1.3,max_vectors=1000)
    states=result['states']
    assert len(states)==7
    assert states[0]['energy_lab_at']==1.
    expected=math.sqrt(.45**2+(2*math.pi/16)**2)+math.sqrt(.55**2+(2*math.pi/16)**2)
    assert all(row['energy_lab_at']==pytest.approx(expected) for row in states[1:])
    assert all(tuple(a+b for a,b in zip(row['n1'],row['n2']))==(0,0,0) for row in states)


def test_moving_mass_exchange_preserves_labelled_states():
    def run(a,b):
        return free_two_body_states(mass1_at=a,mass2_at=b,frame=LatticeFrame(16,1.,(1,1,0)),
                                    energy_max_at=1.5,max_vectors=5000)['states']
    original=run(.45,.55);exchanged=run(.55,.45)
    assert len(original)>0
    assert {(r['n1'],r['n2']):r['energy_lab_at'] for r in original}=={
        (r['n2'],r['n1']):r['energy_lab_at'] for r in exchanged}


def test_budget_and_below_threshold():
    args=dict(mass1_at=.45,mass2_at=.55,frame=LatticeFrame(16,1.))
    with pytest.raises(ValueError,match='budget'):
        free_two_body_states(**args,energy_max_at=1.3,max_vectors=1)
    assert free_two_body_states(**args,energy_max_at=.9,max_vectors=1)['states']==[]


@pytest.mark.parametrize('masses',[(.5,.5),(.45,.55)])
def test_zero_momentum_at_exact_threshold(masses):
    result=free_two_body_states(mass1_at=masses[0],mass2_at=masses[1],frame=LatticeFrame(16,1.),
                                energy_max_at=1.,max_vectors=27)
    assert result['states']==[{'n1':(0,0,0),'n2':(0,0,0),'energy_lab_at':1.}]
